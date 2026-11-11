"""JV-Link (JRA-VAN Data Lab.) からデータを取得して生ファイルに保存する.

Windows + JV-Link インストール済み + pywin32 が必要。
JV-Link は 32bit COM なので 32bit 版 Python で実行すること。
このモジュールは標準ライブラリと win32com だけに依存するので、
学習用の 64bit 環境とは別に、最小構成の 32bit Python から単体で実行できる:

    py -3-32 -m keiba.jvlink_client setup --from 20150101
    py -3-32 -m keiba.jvlink_client update
    py -3-32 -m keiba.jvlink_client week
    py -3-32 -m keiba.jvlink_client weights --date 20261115

保存形式: JV-Data の 1 レコード = 1 行 (Shift-JIS, CRLF)。
JVRead は Unicode 文字列を返すが、レコードのバイト位置は Shift-JIS 基準なので
cp932 に戻して保存する。
"""

from __future__ import annotations

import argparse
import codecs
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# 予想に使うレコード種別 (RACE データ種別には他にオッズ・払戻等も含まれる)
DEFAULT_RECORD_TYPES = ("RA", "SE", "HR", "WH")

# JVOpen option
OPT_NORMAL = 1  # 通常データ (差分)
OPT_THIS_WEEK = 2  # 今週データ (出馬表など)
OPT_SETUP = 3  # セットアップ (ダイアログあり)
OPT_SETUP_NODIALOG = 4  # セットアップ (ダイアログなし)


def _fullwidth_replace(err: UnicodeEncodeError):
    # cp932 に戻せない文字 (外字など) は元が 2 バイト文字なので全角 "？" で置換し、
    # 固定長レコードのバイト位置がずれないようにする。
    return ("？" * (err.end - err.start), err.end)


codecs.register_error("jv_fullwidth", _fullwidth_replace)


def to_sjis(text: str) -> bytes:
    return text.encode("cp932", errors="jv_fullwidth")


class JVLinkError(RuntimeError):
    pass


class JVLinkClient:
    def __init__(self, sid: str = "UNKNOWN"):
        import win32com.client  # noqa: PLC0415  (Windows 専用)

        self.jv = win32com.client.Dispatch("JVDTLab.JVLink")
        ret = self.jv.JVInit(sid)
        if ret != 0:
            raise JVLinkError(f"JVInit failed: {ret}")

    # ------------------------------------------------------------- 蓄積系
    def open(self, dataspec: str, fromtime: str, option: int) -> tuple[int, int, str]:
        res = self.jv.JVOpen(dataspec, fromtime, option, 0, 0, "")
        # pywin32 では出力引数がタプルで返る: (戻り値, 読込ファイル数, DL ファイル数, 最新タイムスタンプ)
        ret, readcount, dlcount, last_ts = res
        if ret == -1:  # 該当データなし
            return 0, 0, last_ts or fromtime
        if ret < 0:
            raise JVLinkError(f"JVOpen({dataspec}, {fromtime}, {option}) failed: {ret}")
        log.info("JVOpen %s from=%s option=%d read=%d download=%d", dataspec, fromtime, option, readcount, dlcount)
        self._wait_download(dlcount)
        return readcount, dlcount, last_ts

    def _wait_download(self, dlcount: int) -> None:
        if dlcount <= 0:
            return
        while True:
            status = self.jv.JVStatus()
            if status < 0:
                raise JVLinkError(f"JVStatus error: {status}")
            log.info("download %d/%d", status, dlcount)
            if status >= dlcount:
                return
            time.sleep(1.0)

    # ------------------------------------------------------------- 速報系
    def rt_open(self, dataspec: str, key: str) -> bool:
        ret = self.jv.JVRTOpen(dataspec, key)
        if ret == -1:
            return False
        if ret < 0:
            raise JVLinkError(f"JVRTOpen({dataspec}, {key}) failed: {ret}")
        return True

    # ------------------------------------------------------------- 読込
    def read_all(self, out_path: Path, record_types=DEFAULT_RECORD_TYPES) -> int:
        """オープン中のデータを全て読み, 対象レコードを out_path に保存する."""
        wanted = set(record_types)
        n = 0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as fh:
            while True:
                ret, buff, _size, fname = self.jv.JVRead("", 110000, "")
                if ret > 0:
                    if buff[:2] in wanted:
                        data = to_sjis(buff)
                        if not data.endswith(b"\r\n"):
                            data = data.rstrip(b"\r\n") + b"\r\n"
                        fh.write(data)
                        n += 1
                    continue
                if ret == 0:  # 全ファイル読込終了
                    break
                if ret == -1:  # ファイル切り替わり
                    continue
                if ret == -3:  # ダウンロード中
                    time.sleep(0.5)
                    continue
                if ret in (-402, -403):  # ファイル破損 -> 削除して次へ
                    log.warning("broken file %s (%d). deleting", fname, ret)
                    self.jv.JVFiledelete(fname)
                    continue
                raise JVLinkError(f"JVRead failed: {ret} ({fname})")
        log.info("saved %d records -> %s", n, out_path)
        return n

    def close(self) -> None:
        self.jv.JVClose()


# ----------------------------------------------------------------- CLI

def _state_path(data_dir: Path) -> Path:
    return data_dir / "jvlink_state.json"


def _load_state(data_dir: Path) -> dict:
    p = _state_path(data_dir)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save_state(data_dir: Path, state: dict) -> None:
    _state_path(data_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _out_file(data_dir: Path, tag: str) -> Path:
    return data_dir / "raw" / f"{tag}_{datetime.now():%Y%m%d_%H%M%S}.jv"


def fetch_stored(data_dir: Path, dataspec: str, fromtime: str, option: int, sid: str) -> Path:
    client = JVLinkClient(sid)
    try:
        _, _, last_ts = client.open(dataspec, fromtime, option)
        out = _out_file(data_dir, f"{dataspec}_opt{option}")
        client.read_all(out)
    finally:
        client.close()
    if option != OPT_THIS_WEEK and last_ts:
        state = _load_state(data_dir)
        state[dataspec] = last_ts
        _save_state(data_dir, state)
    return out


def fetch_realtime(data_dir: Path, dataspec: str, key: str, sid: str) -> Path | None:
    client = JVLinkClient(sid)
    try:
        if not client.rt_open(dataspec, key):
            log.info("no realtime data for %s %s", dataspec, key)
            return None
        out = _out_file(data_dir, f"RT{dataspec}_{key}")
        client.read_all(out)
        return out
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="JV-Link からデータを取得する (Windows 専用)")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--sid", default="UNKNOWN", help="JVInit に渡すソフトウェア ID")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("setup", help="過去データの一括取得 (初回)")
    p.add_argument("--from", dest="fromdate", default="20150101", help="YYYYMMDD")
    p.add_argument("--dialog", action="store_true", help="JV-Link のセットアップダイアログを表示する")

    sub.add_parser("update", help="前回取得以降の差分を取得")
    sub.add_parser("week", help="今週データ (出馬表) を取得")

    p = sub.add_parser("weights", help="当日の馬体重 (速報) を取得")
    p.add_argument("--date", required=True, help="YYYYMMDD または レースキー YYYYMMDDJJKKHHRR")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.data_dir.mkdir(parents=True, exist_ok=True)

    if args.cmd == "setup":
        opt = OPT_SETUP if args.dialog else OPT_SETUP_NODIALOG
        fetch_stored(args.data_dir, "RACE", args.fromdate + "000000", opt, args.sid)
    elif args.cmd == "update":
        state = _load_state(args.data_dir)
        if "RACE" not in state:
            raise SystemExit("前回の取得記録がありません。先に setup を実行してください。")
        fetch_stored(args.data_dir, "RACE", state["RACE"], OPT_NORMAL, args.sid)
    elif args.cmd == "week":
        since = (datetime.now() - timedelta(days=7)).strftime("%Y%m%d") + "000000"
        fetch_stored(args.data_dir, "RACE", since, OPT_THIS_WEEK, args.sid)
    elif args.cmd == "weights":
        fetch_realtime(args.data_dir, "0B11", args.date, args.sid)


if __name__ == "__main__":
    main()
