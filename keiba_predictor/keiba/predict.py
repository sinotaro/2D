"""出馬表 (結果未確定のレース) に対する予想の作成と表示."""

from __future__ import annotations

import logging

import pandas as pd

from .model import Models, predict

log = logging.getLogger(__name__)

JYO_NAME = {"01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
            "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉"}


def predict_races(models: Models, feat: pd.DataFrame, date: str, race_ids=None, jyo=None) -> pd.DataFrame:
    target = feat[feat["date"] == pd.Timestamp(date)]
    if race_ids:
        target = target[target["race_id"].isin(race_ids)]
    if jyo:
        target = target[target["jyo"].isin(jyo)]
    if target.empty:
        return target
    missing_weight = target["horse_weight"].isna().mean()
    if missing_weight > 0.5:
        log.warning("馬体重が未発表の馬が %.0f%% あります (発表後に weights を取得すると精度が上がります)",
                    missing_weight * 100)
    out = predict(models, target)
    cols = ["race_id", "date", "jyo", "race_no", "race_name", "pred_rank", "umaban", "horse_name",
            "jockey_name", "win_prob", "top3_prob", "score", "finish_pos", "odds", "ninki"]
    return out[cols]


def format_race(g: pd.DataFrame, top: int = 0) -> str:
    head = g.iloc[0]
    jyo = JYO_NAME.get(head["jyo"], head["jyo"])
    lines = [f"■ {head['date']:%Y-%m-%d} {jyo}{int(head['race_no'])}R {head['race_name']}  ({head['race_id']})",
             " 予想  馬番  馬名                  騎手        勝率   複勝率"
             + ("   着順" if g["finish_pos"].notna().any() else "")]
    rows = g.sort_values("pred_rank")
    if top:
        rows = rows.head(top)
    marks = {1: "◎", 2: "○", 3: "▲", 4: "△", 5: "☆"}
    for _, r in rows.iterrows():
        res = f"   {int(r['finish_pos']):>3}" if pd.notna(r["finish_pos"]) else ""
        lines.append(
            f" {marks.get(r['pred_rank'], '  ')}{r['pred_rank']:>2}  {int(r['umaban']):>4}  "
            f"{_pad(r['horse_name'], 20)}  {_pad(r['jockey_name'], 10)}  "
            f"{r['win_prob'] * 100:5.1f}%  {r['top3_prob'] * 100:5.1f}%{res}"
        )
    return "\n".join(lines)


def _pad(text, width: int) -> str:
    """全角を 2 桁として左寄せ."""
    text = str(text or "")
    w = sum(2 if ord(ch) > 0xFF else 1 for ch in text)
    return text + " " * max(0, width - w)
