"""LightGBM による着順予想モデルの学習・評価・推論.

3 つのモデルを学習する:
  ranker : LambdaRank。レース内の並び順 (予想着順) を決める
  win    : 1 着になる確率 (二値分類, レース内で合計 1 に正規化)
  top3   : 3 着以内に入る確率 (二値分類, レース内で合計 3 に正規化)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from .features import CATEGORICAL, FEATURES, relevance_label

log = logging.getLogger(__name__)

PARAMS_COMMON = dict(
    learning_rate=0.05, num_leaves=63, min_child_samples=50, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42,
)
RANK_PARAMS = dict(objective="lambdarank", metric="ndcg", eval_at=[1, 3], lambdarank_truncation_level=5, **PARAMS_COMMON)
BIN_PARAMS = dict(objective="binary", metric="binary_logloss", **PARAMS_COMMON)


@dataclass
class Models:
    ranker: lgb.Booster
    win: lgb.Booster
    top3: lgb.Booster
    features: list[str]

    def save(self, model_dir: Path, meta: dict | None = None) -> None:
        model_dir.mkdir(parents=True, exist_ok=True)
        for name in ("ranker", "win", "top3"):
            getattr(self, name).save_model(str(model_dir / f"{name}.txt"))
        info = {"features": self.features, **(meta or {})}
        (model_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: Path) -> "Models":
        meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
        boosters = {n: lgb.Booster(model_file=str(model_dir / f"{n}.txt")) for n in ("ranker", "win", "top3")}
        return cls(features=meta["features"], **boosters)


def training_rows(feat: pd.DataFrame) -> pd.DataFrame:
    """結果が確定しているレースだけ (全頭の着順が揃っているもの)."""
    df = feat[feat["has_result"]]
    complete = df.groupby("race_id")["finish_pos"].transform("count") > 0
    df = df[complete & (df["n_runners"] >= 2)]
    return df.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)


def _dataset(df: pd.DataFrame, label: pd.Series, group: bool, reference=None) -> lgb.Dataset:
    ds = lgb.Dataset(
        df[FEATURES], label=label, categorical_feature=[c for c in CATEGORICAL if c in FEATURES],
        free_raw_data=False, reference=reference,
    )
    if group:
        ds.set_group(df.groupby("race_id", sort=False).size().to_numpy())
    return ds


def _train_one(params, train_df, valid_df, label_fn, group, num_rounds, early_stop) -> lgb.Booster:
    dtrain = _dataset(train_df, label_fn(train_df), group)
    valid_sets, callbacks = [], [lgb.log_evaluation(100)]
    if valid_df is not None and len(valid_df):
        valid_sets = [_dataset(valid_df, label_fn(valid_df), group, reference=dtrain)]
        callbacks.append(lgb.early_stopping(early_stop, verbose=False))
    booster = lgb.train(params, dtrain, num_boost_round=num_rounds, valid_sets=valid_sets, callbacks=callbacks)
    return booster


def train(feat: pd.DataFrame, valid_from: str | None = None, num_rounds: int = 2000,
          early_stop: int = 100, min_history_date: str | None = None) -> tuple[Models, dict]:
    """valid_from 以降を検証期間として学習する.

    検証で決まった木の本数を使い, 最後に全期間で学習し直したモデルを返す。
    """
    df = training_rows(feat)
    if min_history_date:
        # データ先頭付近は過去成績が空なので学習から外す
        df = df[df["date"] >= pd.Timestamp(min_history_date)]
    labels = {
        "ranker": lambda d: relevance_label(d["finish_pos"]),
        "win": lambda d: (d["finish_pos"] == 1).astype(int),
        "top3": lambda d: (d["finish_pos"] <= 3).astype(int),
    }
    params = {"ranker": RANK_PARAMS, "win": BIN_PARAMS, "top3": BIN_PARAMS}
    report: dict = {"n_train_races": int(df["race_id"].nunique())}

    best_iters = {}
    if valid_from:
        vf = pd.Timestamp(valid_from)
        tr, va = df[df["date"] < vf], df[df["date"] >= vf]
        if tr.empty or va.empty:
            raise ValueError(f"valid_from={valid_from} で学習/検証期間のどちらかが空です")
        boosters = {}
        for name in labels:
            log.info("training %s (validation from %s)", name, valid_from)
            boosters[name] = _train_one(params[name], tr, va, labels[name], name == "ranker", num_rounds, early_stop)
            best_iters[name] = max(boosters[name].best_iteration, 50)
        m = Models(features=list(FEATURES), **boosters)
        report["validation"] = evaluate(m, va)
        report["valid_from"] = valid_from
        report["best_iterations"] = best_iters
        log.info("validation: %s", json.dumps(report["validation"], ensure_ascii=False))

    final = {}
    for name in labels:
        rounds = best_iters.get(name, 300 if valid_from is None else num_rounds)
        log.info("training final %s (%d rounds, all data)", name, rounds)
        final[name] = _train_one(params[name], df, None, labels[name], name == "ranker", rounds, early_stop)
    report["train_until"] = str(df["date"].max().date())
    return Models(features=list(FEATURES), **final), report


# ----------------------------------------------------------------- 推論

def _normalize_in_race(race_id: pd.Series, p: np.ndarray, total: float) -> np.ndarray:
    s = pd.Series(p, index=race_id.index).groupby(race_id).transform("sum").to_numpy()
    return np.clip(p / np.where(s > 0, s, 1) * total, 0, 1)


def predict(models: Models, feat: pd.DataFrame) -> pd.DataFrame:
    X = feat[models.features]
    out = feat.copy()
    out["score"] = models.ranker.predict(X)
    out["win_prob"] = _normalize_in_race(feat["race_id"], models.win.predict(X), 1.0)
    top3_total = np.minimum(3, feat.groupby("race_id")["umaban"].transform("count").to_numpy())
    raw_top3 = models.top3.predict(X)
    s = pd.Series(raw_top3, index=feat.index).groupby(feat["race_id"]).transform("sum").to_numpy()
    out["top3_prob"] = np.clip(raw_top3 / np.where(s > 0, s, 1) * top3_total, 0, 1)
    out["pred_rank"] = out.groupby("race_id")["score"].rank(ascending=False, method="first").astype(int)
    return out.sort_values(["date", "race_id", "pred_rank"]).reset_index(drop=True)


# ----------------------------------------------------------------- 評価

def evaluate(models: Models, df: pd.DataFrame) -> dict:
    p = predict(models, df)
    p = p[p["finish_pos"].notna()]
    top = p[p["pred_rank"] == 1]
    n_races = int(p["race_id"].nunique())

    ndcg1, ndcg3 = [], []
    for _, g in p.groupby("race_id"):
        if len(g) < 2:
            continue
        rel = relevance_label(g["finish_pos"]).to_numpy()[None, :]
        sc = g["score"].to_numpy()[None, :]
        ndcg1.append(ndcg_score(rel, sc, k=1))
        ndcg3.append(ndcg_score(rel, sc, k=3))

    # 1 番人気 (ベースライン)
    fav = p[p["ninki"] == 1]
    res = {
        "races": n_races,
        "ndcg@1": float(np.mean(ndcg1)),
        "ndcg@3": float(np.mean(ndcg3)),
        "top1_win_rate": float((top["finish_pos"] == 1).mean()),
        "top1_top3_rate": float((top["finish_pos"] <= 3).mean()),
        "favorite_win_rate": float((fav["finish_pos"] == 1).mean()) if len(fav) else None,
        "random_win_rate": float((1 / p.groupby("race_id")["umaban"].transform("count")).groupby(p["race_id"]).first().mean()),
        "spearman": float(np.nanmean([g["pred_rank"].corr(g["finish_pos"], method="spearman")
                                      for _, g in p.groupby("race_id") if len(g) >= 3])),
        "logloss_win": float(_logloss((p["finish_pos"] == 1).to_numpy(), p["win_prob"].to_numpy())),
    }
    # 単勝回収率 (払戻は確定オッズで近似。オッズは評価にのみ使い特徴量には使っていない)
    odds_ok = top["odds"].notna()
    if odds_ok.any():
        t = top[odds_ok]
        res["top1_tansho_return"] = float(((t["finish_pos"] == 1) * t["odds"]).sum() / len(t))
        ev = p[p["odds"].notna() & (p["win_prob"] * p["odds"] > 1.2)]
        res["ev_bets"] = int(len(ev))
        res["ev_tansho_return"] = float(((ev["finish_pos"] == 1) * ev["odds"]).sum() / len(ev)) if len(ev) else None
    return res


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def feature_importance(models: Models, top: int = 30) -> pd.DataFrame:
    imp = pd.DataFrame({
        "feature": models.features,
        "ranker_gain": models.ranker.feature_importance("gain"),
        "win_gain": models.win.feature_importance("gain"),
    })
    imp["ranker_gain"] /= imp["ranker_gain"].sum()
    imp["win_gain"] /= imp["win_gain"].sum()
    return imp.sort_values("ranker_gain", ascending=False).head(top).reset_index(drop=True)
