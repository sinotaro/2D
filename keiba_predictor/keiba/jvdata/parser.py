"""JV-Data 固定長レコード (Shift-JIS バイト列) のパーサ."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from . import spec

log = logging.getLogger(__name__)

ENCODING = "cp932"
_STRIP = " 　\x00"


def _decode(raw: bytes) -> str:
    return raw.decode(ENCODING, errors="replace")


def _convert(raw: bytes, kind: str):
    text = _decode(raw)
    if kind == "raw":
        return text
    text = text.strip(_STRIP)
    if kind == "str":
        return text
    if kind == "int":
        return int(text) if text.isdigit() else None
    raise ValueError(f"unknown field kind: {kind}")


def decode_fields(line: bytes, fields: list[spec.Field], offset: int = 0) -> dict:
    out = {}
    for f in fields:
        s = f.slice
        out[f.name] = _convert(line[s.start + offset : s.stop + offset], f.kind)
    return out


# ---------------------------------------------------------------- 値の正規化

def make_race_id(year: str, monthday: str, jyo: str, kai: int, nichi: int, race_no: int) -> str:
    """YYYYMMDD + 場コード2 + 回2 + 日目2 + R2 の 16 桁."""
    return f"{year}{monthday}{jyo}{kai or 0:02d}{nichi or 0:02d}{race_no or 0:02d}"


def parse_race_time(text: str) -> float | None:
    """走破タイム "1345" -> 94.5 秒."""
    text = text.strip()
    if len(text) != 4 or not text.isdigit() or text == "0000":
        return None
    return int(text[0]) * 60 + int(text[1:3]) + int(text[3]) / 10


def parse_tenths(text: str, invalid: tuple[str, ...] = ()) -> float | None:
    """0.1 単位の数字 ("345" -> 34.5, "0123" -> 12.3)."""
    text = text.strip()
    if not text.isdigit() or int(text) == 0 or text in invalid:
        return None
    return int(text) / 10


def parse_signed_tenths(text: str) -> float | None:
    """タイム差 "+005" -> 0.5, "-003" -> -0.3."""
    text = text.strip()
    if not text:
        return None
    sign = -1 if text[0] == "-" else 1
    digits = text.lstrip("+- ")
    if not digits.isdigit() or digits == "999":
        return None
    return sign * int(digits) / 10


def parse_weight(weight: str, sign: str, diff: str) -> tuple[int | None, int | None]:
    """馬体重と増減. 999 は計量不能, 000/空白は未発表."""
    w = weight.strip()
    hw = int(w) if w.isdigit() and w not in ("000", "999") else None
    d = diff.strip()
    if hw is None or not d.isdigit() or d == "999":
        return hw, None
    return hw, (-1 if sign == "-" else 1) * int(d)


# ---------------------------------------------------------------- レコード

def parse_ra(line: bytes) -> dict:
    r = decode_fields(line, spec.RA)
    return {
        "race_id": make_race_id(r["year"], r["monthday"], r["jyo"], r["kai"], r["nichi"], r["race_no"]),
        "data_kubun": r["data_kubun"],
        "made_date": r["made_date"],
        "date": f"{r['year']}-{r['monthday'][:2]}-{r['monthday'][2:]}",
        "jyo": r["jyo"],
        "kai": r["kai"],
        "nichi": r["nichi"],
        "race_no": r["race_no"],
        "race_name": r["race_name"] or r["race_name_short"],
        "grade": r["grade"].strip(),
        "syubetsu": r["syubetsu"],
        "kigo": r["kigo"],
        "juryo": r["juryo"],
        "jyoken": r["jyoken_young"],
        "distance": r["distance"],
        "track_code": r["track_code"],
        "course_kubun": r["course_kubun"].strip(),
        "prize1": r["honsyokin1"],
        "start_time": r["start_time"],
        "toroku_tosu": r["toroku_tosu"],
        "syusso_tosu": r["syusso_tosu"],
        "weather": r["weather"].strip(),
        "turf_going": r["turf_going"].strip(),
        "dirt_going": r["dirt_going"].strip(),
        "race_first3f": parse_tenths(f"{r['haron_first3'] or 0:03d}", ("999",)),
        "race_last3f": parse_tenths(f"{r['haron_last3'] or 0:03d}", ("999",)),
    }


def parse_se(line: bytes) -> dict:
    r = decode_fields(line, spec.SE)
    hw, wd = parse_weight(r["horse_weight"], r["weight_sign"], r["weight_diff"])
    return {
        "race_id": make_race_id(r["year"], r["monthday"], r["jyo"], r["kai"], r["nichi"], r["race_no"]),
        "umaban": r["umaban"],
        "data_kubun": r["data_kubun"],
        "made_date": r["made_date"],
        "wakuban": r["wakuban"],
        "horse_id": r["horse_id"].strip(),
        "horse_name": r["horse_name"],
        "sex": r["sex"].strip(),
        "age": r["age"],
        "tozai": r["tozai"].strip(),
        "trainer_id": r["trainer_id"].strip(),
        "trainer_name": r["trainer_name"],
        "owner_id": r["owner_id"].strip(),
        "jockey_id": r["jockey_id"].strip(),
        "jockey_name": r["jockey_name"],
        "minarai": r["minarai"].strip(),
        "futan": r["futan"] / 10 if r["futan"] else None,
        "blinker": r["blinker"].strip(),
        "horse_weight": hw,
        "weight_diff": wd,
        "ijou": r["ijou"].strip() or "0",
        "finish_pos": r["kakutei_jyuni"] or None,
        "time": parse_race_time(r["time"]),
        "time_diff": parse_signed_tenths(r["time_diff"]),
        "corner1": r["corner1"] or None,
        "corner2": r["corner2"] or None,
        "corner3": r["corner3"] or None,
        "corner4": r["corner4"] or None,
        "odds": parse_tenths(r["odds"], ("9999",)),
        "ninki": r["ninki"] or None,
        "prize": r["honsyokin"],
        "last3f": parse_tenths(r["haron_last3"], ("999",)),
        "last4f": parse_tenths(r["haron_last4"], ("999",)),
        "kyakusitu": r["kyakusitu"].strip(),
    }


def parse_wh(line: bytes) -> list[dict]:
    h = decode_fields(line, spec.WH_HEADER)
    race_id = make_race_id(h["year"], h["monthday"], h["jyo"], h["kai"], h["nichi"], h["race_no"])
    out = []
    for i in range(spec.WH_ITEM_COUNT):
        offset = spec.WH_ITEM_START - 1 + i * spec.WH_ITEM_SIZE
        item = decode_fields(line, spec.WH_ITEM, offset=offset)
        if not item["umaban"]:
            continue
        hw, wd = parse_weight(item["horse_weight"], item["weight_sign"], item["weight_diff"])
        out.append({
            "race_id": race_id,
            "umaban": item["umaban"],
            "announced": h["announced"],
            "horse_weight": hw,
            "weight_diff": wd,
        })
    return out


PARSERS = {"RA": parse_ra, "SE": parse_se, "WH": parse_wh}


def iter_lines(path: Path) -> Iterator[bytes]:
    """生データファイル (CRLF 区切りの Shift-JIS) を 1 レコードずつ返す."""
    with open(path, "rb") as fh:
        for line in fh:
            line = line.rstrip(b"\r\n")
            if line:
                yield line


def iter_records(path: Path) -> Iterator[tuple[str, dict | list[dict]]]:
    """(レコード種別, パース結果) を返す. 未対応の種別は読み飛ばす."""
    for n, line in enumerate(iter_lines(path), 1):
        rtype = line[:2].decode("ascii", errors="replace")
        parser = PARSERS.get(rtype)
        if parser is None:
            continue
        expected = spec.RECORD_LENGTHS[rtype] - 2  # CRLF を除く
        if len(line) != expected:
            log.warning("%s:%d %s レコード長が %d (期待値 %d)。スキップ", path.name, n, rtype, len(line), expected)
            continue
        yield rtype, parser(line)
