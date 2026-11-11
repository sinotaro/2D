"""コマンドライン:

    python -m keiba load                       生データ (data/raw/*.jv) を DB に取り込む
    python -m keiba train [--valid-from DATE]  モデルを学習・評価して models/ に保存
    python -m keiba predict --date 2026-11-15  指定日の出馬表から着順を予想
    python -m keiba demo                       架空データで一連の流れを試す

JV-Link からの取得は Windows (32bit Python) で:
    python -m keiba.jvlink_client setup / update / week / weights
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

log = logging.getLogger("keiba")


def _paths(args) -> tuple[Path, Path, Path]:
    data = Path(args.data_dir)
    return data / "raw", data / "keiba.sqlite", Path(args.model_dir)


def cmd_load(args) -> None:
    from . import db

    raw_dir, db_path, _ = _paths(args)
    conn = db.connect(db_path)
    n = db.load_raw_dir(conn, raw_dir, force=args.force)
    races = conn.execute("SELECT COUNT(*) FROM races").fetchone()[0]
    entries = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    print(f"取込 {n} レコード / DB: レース {races}, 出走 {entries}")


def _features(args):
    from . import db, features

    _, db_path, _ = _paths(args)
    conn = db.connect(db_path)
    raw = db.read_dataset(conn)
    if raw.empty:
        sys.exit("DB にデータがありません。先に load を実行してください。")
    log.info("building features for %d rows", len(raw))
    return features.build_features(raw)


def cmd_train(args) -> None:
    import pandas as pd

    from . import model

    _, _, model_dir = _paths(args)
    feat = _features(args)
    dates = feat.loc[feat["has_result"], "date"]
    first, last = dates.min(), dates.max()
    skip = args.skip_days if args.skip_days is not None else min(365, int((last - first).days * 0.1))
    min_date = (first + pd.Timedelta(days=skip)).date().isoformat()
    valid_from = args.valid_from
    if valid_from is None and not args.no_valid:
        valid_from = (first + (last - first) * 0.8).date().isoformat()

    models, report = model.train(feat, valid_from=valid_from, min_history_date=min_date, num_rounds=args.rounds)
    report["min_history_date"] = min_date
    models.save(model_dir, meta=report)

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("\n重要な特徴量 (ranker, gain 比):")
    print(model.feature_importance(models, 20).to_string(index=False))
    print(f"\nモデル保存先: {model_dir}")


def cmd_predict(args) -> None:
    from . import model
    from .predict import format_race, predict_races

    _, _, model_dir = _paths(args)
    models = model.Models.load(model_dir)
    feat = _features(args)
    pred = predict_races(models, feat, date=args.date, race_ids=args.race_id, jyo=args.jyo)
    if pred.empty:
        sys.exit(f"{args.date} の出走データがありません (出馬表を取得・取込済みか確認してください)")
    for _, g in pred.groupby("race_id", sort=False):
        print(format_race(g, top=args.top))
        print()
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out, index=False, encoding="utf-8-sig")
        print(f"保存: {out}")


def cmd_demo(args) -> None:
    from .synthetic import write_synthetic

    data_dir = Path(args.data_dir)
    _, _, target = write_synthetic(data_dir / "raw", seed=args.seed, weeks=args.weeks)
    print(f"架空データを {data_dir / 'raw'} に生成しました。予想対象日: {target}")
    cmd_load(argparse.Namespace(**{**vars(args), "force": True}))
    cmd_train(argparse.Namespace(**{**vars(args), "valid_from": None, "no_valid": False, "skip_days": None}))
    cmd_predict(argparse.Namespace(**{**vars(args), "date": target, "race_id": None, "jyo": None, "out": None}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="keiba", description="JV-Link データによる競馬着順予想")
    ap.add_argument("--data-dir", default=os.environ.get("KEIBA_DATA_DIR", "data"))
    ap.add_argument("--model-dir", default=os.environ.get("KEIBA_MODEL_DIR", "models"))
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("load", help="生データを DB に取り込む")
    p.add_argument("--force", action="store_true", help="取込済みファイルも再取込")
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("train", help="モデルを学習")
    p.add_argument("--valid-from", help="この日以降を検証期間にする (既定: 期間の後ろ 20%%)")
    p.add_argument("--no-valid", action="store_true", help="検証せず全期間で学習")
    p.add_argument("--skip-days", type=int, help="データ先頭から学習に使わない日数 (過去成績が無いため)")
    p.add_argument("--rounds", type=int, default=2000)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="着順を予想")
    p.add_argument("--date", required=True, help="YYYY-MM-DD")
    p.add_argument("--race-id", nargs="*", help="16 桁のレース ID で絞り込み")
    p.add_argument("--jyo", nargs="*", help="場コードで絞り込み (05=東京 06=中山 08=京都 09=阪神 ...)")
    p.add_argument("--top", type=int, default=0, help="各レース上位 N 頭だけ表示 (0=全頭)")
    p.add_argument("--out", help="CSV 出力先")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("demo", help="架空データで動作確認")
    p.add_argument("--weeks", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--rounds", type=int, default=2000)
    p.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
