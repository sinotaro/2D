"""特徴量生成.

原則: 各出走行の特徴量は「そのレースの発走前に分かる情報」だけで作る。
  * 当該レースの条件 (距離・馬場・頭数・枠・斤量・馬体重 など)
  * そのレースより前の日付のレース結果から集計した成績
騎手・調教師の成績は同じ日の他レースの結果も混ざらないよう、日単位で集計して
前日までの累積を使う。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- 定数

NOT_STARTED = {"1", "2", "3"}  # 出走取消, 発走除外, 競走除外
DNF = {"4", "5"}  # 競走中止, 失格

GRADE_LEVEL = {"A": 6, "F": 6, "B": 5, "G": 5, "C": 4, "H": 4, "L": 3, "D": 2, "E": 1}
# 最若年条件コード -> クラス
JYOKEN_LEVEL = {"701": 0, "702": 0, "703": 0, "005": 1, "010": 2, "016": 3, "999": 4}

ID_COLS = [
    "race_id", "date", "jyo", "race_no", "race_name", "umaban", "horse_id", "horse_name",
    "jockey_name", "finish_pos", "odds", "ninki",
]

CATEGORICAL = ["jyo_c", "surface_c", "sex_c", "tozai_c", "track_c"]

HIST_K = 5  # 直近何走を見るか
SMOOTH_N = 30  # 騎手・調教師勝率の平滑化 (疑似出走数)


def surface_of(track_code: pd.Series) -> pd.Series:
    code = pd.to_numeric(track_code, errors="coerce")
    return pd.Series(
        np.select([code.between(10, 22), code.between(23, 29), code.between(51, 59)], ["turf", "dirt", "jump"], "other"),
        index=track_code.index,
    )


def distance_bin(distance: pd.Series) -> pd.Series:
    return pd.cut(distance, [0, 1400, 1800, 2200, 10000], labels=[0, 1, 2, 3]).astype(float)


# ----------------------------------------------------------------- 前処理

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """型変換・出走馬への絞り込み・1 走ごとの成績指標を計算."""
    df = df.copy()
    df["ijou"] = df["ijou"].fillna("0").astype(str)
    df = df[~df["ijou"].isin(NOT_STARTED)].copy()
    df = df[df["horse_id"].notna() & (df["horse_id"] != "")]

    df["date"] = pd.to_datetime(df["date"])
    num = ["umaban", "wakuban", "age", "futan", "horse_weight", "weight_diff", "finish_pos", "time",
           "corner4", "odds", "ninki", "last3f", "distance", "prize1", "syusso_tosu", "race_no"]
    for c in num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["surface"] = surface_of(df["track_code"])
    going = np.where(df["surface"] == "dirt", df["dirt_going"], df["turf_going"])
    df["going"] = pd.to_numeric(pd.Series(going, index=df.index), errors="coerce").replace(0, np.nan)
    df["weather_c"] = pd.to_numeric(df["weather"], errors="coerce").replace(0, np.nan)
    df["dist_bin"] = distance_bin(df["distance"])
    df["heavy"] = (df["going"] >= 3).astype(float)

    df = df.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)
    race = df.groupby("race_id")
    df["n_runners"] = race["umaban"].transform("count")

    # ---- 結果系 (発走後に分かる。過去走の集計にのみ使い, そのレース自身の特徴量には使わない)
    finished = df["finish_pos"].notna() & (df["finish_pos"] > 0)
    dnf = df["ijou"].isin(DNF)
    df["has_result"] = finished | dnf
    n1 = (df["n_runners"] - 1).clip(lower=1)
    df["r_pos_ratio"] = np.where(finished, (df["finish_pos"] - 1) / n1, np.where(dnf, 1.0, np.nan))
    df["r_win"] = np.where(df["has_result"], (df["finish_pos"] == 1).astype(float), np.nan)
    df["r_top3"] = np.where(df["has_result"], (df["finish_pos"] <= 3).astype(float), np.nan)
    df["r_behind"] = (df["time"] - race["time"].transform("min")).clip(0, 5).where(finished)
    df["r_last3f_rel"] = (df["last3f"] - race["last3f"].transform("median")).clip(-3, 3).where(finished)
    df["r_corner4_ratio"] = (df["corner4"] / df["n_runners"]).where(finished)
    df["r_logprize"] = np.log1p(df["prize1"].fillna(0))
    df["r_prize_earned"] = np.where(finished & (df["finish_pos"] <= 5), df["r_logprize"], 0.0)
    return df


# ----------------------------------------------------------------- 馬の過去成績

def _lags(g, col: str, k: int) -> np.ndarray:
    return np.column_stack([g[col].shift(i).to_numpy(dtype=float) for i in range(1, k + 1)])


def _nanmean(a: np.ndarray) -> np.ndarray:
    cnt = np.sum(~np.isnan(a), axis=1)
    s = np.nansum(a, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, s / cnt, np.nan)


def _prior_sum(df: pd.DataFrame, keys: list[str], col: str) -> pd.Series:
    """同じ keys のグループ内で, 自分より前の行の col の合計 (自分は含まない)."""
    groups = [df[k] for k in keys]
    cum = df[col].fillna(0).groupby(groups).cumsum()
    # cumsum - 自分 だと浮動小数の誤差に自分の結果が混ざるので, 1 行ずらして厳密に除外する
    return cum.groupby(groups).shift(1).fillna(0)


def add_horse_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["horse_id", "date", "race_id"]).copy()
    g = df.groupby("horse_id", sort=False)

    df["h_runs"] = _prior_sum(df, ["horse_id"], "has_result")
    wins = _prior_sum(df, ["horse_id"], "r_win")
    top3 = _prior_sum(df, ["horse_id"], "r_top3")
    df["h_win_rate"] = (wins / df["h_runs"]).where(df["h_runs"] > 0)
    df["h_top3_rate"] = (top3 / df["h_runs"]).where(df["h_runs"] > 0)
    df["h_prize_sum"] = _prior_sum(df, ["horse_id"], "r_prize_earned")

    for col, name in [("r_pos_ratio", "pos"), ("r_behind", "behind"), ("r_last3f_rel", "l3f"),
                      ("r_corner4_ratio", "c4"), ("r_logprize", "class")]:
        lags = _lags(g, col, HIST_K)
        df[f"h_{name}_avg{HIST_K}"] = _nanmean(lags)
        df[f"h_{name}_last1"] = lags[:, 0]
    df["h_pos_last2"] = g["r_pos_ratio"].shift(2)
    df["h_pos_last3"] = g["r_pos_ratio"].shift(3)
    df["h_pos_best5"] = np.nanmin(np.where(np.isnan(_lags(g, "r_pos_ratio", HIST_K)), np.inf,
                                           _lags(g, "r_pos_ratio", HIST_K)), axis=1)
    df.loc[np.isinf(df["h_pos_best5"]), "h_pos_best5"] = np.nan
    # 着順の改善傾向 (直近 - 2〜5 走前平均)
    df["h_pos_trend"] = df["h_pos_last1"] - _nanmean(_lags(g, "r_pos_ratio", HIST_K)[:, 1:])

    prev_date = g["date"].shift(1)
    df["h_days_since"] = (df["date"] - prev_date).dt.days
    df["h_dist_change"] = df["distance"] - g["distance"].shift(1)
    df["h_futan_change"] = df["futan"] - g["futan"].shift(1)
    df["h_class_change"] = df["r_logprize"] - df["h_class_avg5"]
    df["h_jockey_changed"] = (df["jockey_id"] != g["jockey_id"].shift(1)).astype(float).where(prev_date.notna())
    df["h_surface_changed"] = (df["surface"] != g["surface"].shift(1)).astype(float).where(prev_date.notna())

    # 条件別適性: 同じ条件での過去の出走数と 3 着内率
    for keys, name in [(["surface"], "surf"), (["surface", "dist_bin"], "dist"), (["jyo", "surface"], "course"),
                       (["heavy"], "heavy"), (["jockey_id"], "jockey")]:
        k = ["horse_id"] + keys
        runs = _prior_sum(df, k, "has_result")
        t3 = _prior_sum(df, k, "r_top3")
        df[f"h_{name}_runs"] = runs
        df[f"h_{name}_top3"] = ((t3 + 0.25 * 2) / (runs + 2))  # 少数サンプルを平均方向へ寄せる
    return df


# ----------------------------------------------------------------- 騎手・調教師

def _daily_prior_stats(df: pd.DataFrame, key: str, prefix: str, window_days: int | None = None) -> pd.DataFrame:
    """key ごとに, [date - window_days, date) の (出走数, 勝率, 3着内率) を返す.

    当日の他レースの結果は含めない (日単位で累積してから 1 日ずらす)。
    window_days=None なら全期間。
    """
    daily = (
        df[df["has_result"]].groupby([key, "date"], sort=True)
        .agg(n=("has_result", "size"), w=("r_win", "sum"), t=("r_top3", "sum"))
        .reset_index()
    )
    cols = ["n", "w", "t"]
    daily[cols] = daily.groupby(key)[cols].cumsum()  # その日を含む累積
    cum = daily.rename(columns={"date": "_d"}).sort_values("_d")

    target = df[[key, "date"]].drop_duplicates().sort_values("date")
    before = pd.merge_asof(target, cum, left_on="date", right_on="_d", by=key, allow_exact_matches=False)
    vals = before[cols].fillna(0).to_numpy()
    if window_days is not None:
        start = target.assign(_start=target["date"] - pd.Timedelta(days=window_days)).sort_values("_start")
        old = pd.merge_asof(start, cum, left_on="_start", right_on="_d", by=key, allow_exact_matches=False)
        old = old.set_index([key, "date"]).reindex(pd.MultiIndex.from_frame(before[[key, "date"]]))
        vals = vals - old[cols].fillna(0).to_numpy()

    out = before[[key, "date"]].copy()
    n, w, t = vals[:, 0], vals[:, 1], vals[:, 2]
    out[f"{prefix}_n"] = n
    out[f"{prefix}_win"] = (w + SMOOTH_N * 0.07) / (n + SMOOTH_N)
    out[f"{prefix}_top3"] = (t + SMOOTH_N * 0.21) / (n + SMOOTH_N)
    return out


def add_people_stats(df: pd.DataFrame) -> pd.DataFrame:
    for key, prefix, window in [("jockey_id", "j", None), ("jockey_id", "j365", 365),
                                ("trainer_id", "t", None), ("trainer_id", "t365", 365)]:
        df = df.merge(_daily_prior_stats(df, key, prefix, window), on=[key, "date"], how="left")
    return df


# ----------------------------------------------------------------- 当該レースの条件

def add_race_features(df: pd.DataFrame) -> pd.DataFrame:
    race = df.groupby("race_id")
    df["umaban_ratio"] = df["umaban"] / df["n_runners"]
    df["futan_diff"] = df["futan"] - race["futan"].transform("mean")
    df["futan_per_weight"] = df["futan"] / df["horse_weight"]
    df["weight_vs_race"] = df["horse_weight"] - race["horse_weight"].transform("mean")
    df["grade_level"] = df["grade"].map(GRADE_LEVEL).fillna(0)
    df["jyoken_level"] = df["jyoken"].map(JYOKEN_LEVEL)
    df["is_handicap"] = (df["juryo"] == "1").astype(float)
    df["minarai_c"] = pd.to_numeric(df["minarai"], errors="coerce").fillna(0)
    df["blinker_c"] = (df["blinker"] == "1").astype(float)
    df["sex_c"] = pd.to_numeric(df["sex"], errors="coerce")
    df["tozai_c"] = pd.to_numeric(df["tozai"], errors="coerce")
    df["jyo_c"] = pd.to_numeric(df["jyo"], errors="coerce")
    df["track_c"] = pd.to_numeric(df["track_code"], errors="coerce")
    df["surface_c"] = df["surface"].map({"turf": 0, "dirt": 1, "jump": 2, "other": 3})
    df["month"] = df["date"].dt.month
    df["race_logprize"] = df["r_logprize"]
    return df


RELATIVE_COLS = [
    "h_pos_avg5", "h_behind_avg5", "h_l3f_avg5", "h_top3_rate", "h_prize_sum", "h_class_avg5",
    "h_dist_top3", "j_win", "j365_top3", "t365_top3", "futan", "horse_weight", "age",
]


def add_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """同じレースの他馬と比べた位置 (ランキング学習には特に効く)."""
    race = df.groupby("race_id")
    for c in RELATIVE_COLS:
        df[f"{c}_rrank"] = race[c].rank(pct=True, na_option="keep")
        df[f"{c}_rdiff"] = df[c] - race[c].transform("mean")
    return df


# ----------------------------------------------------------------- まとめ

FEATURES = [
    # 当該レース
    "umaban", "wakuban", "umaban_ratio", "n_runners", "age", "sex_c", "tozai_c", "futan", "futan_diff",
    "futan_per_weight", "horse_weight", "weight_diff", "weight_vs_race", "minarai_c", "blinker_c",
    "distance", "surface_c", "track_c", "jyo_c", "going", "weather_c", "grade_level", "jyoken_level",
    "is_handicap", "race_logprize", "month", "race_no",
    # 馬の過去成績
    "h_runs", "h_win_rate", "h_top3_rate", "h_prize_sum",
    "h_pos_avg5", "h_pos_last1", "h_pos_last2", "h_pos_last3", "h_pos_best5", "h_pos_trend",
    "h_behind_avg5", "h_behind_last1", "h_l3f_avg5", "h_l3f_last1", "h_c4_avg5", "h_c4_last1",
    "h_class_avg5", "h_class_last1", "h_class_change",
    "h_days_since", "h_dist_change", "h_futan_change", "h_jockey_changed", "h_surface_changed",
    "h_surf_runs", "h_surf_top3", "h_dist_runs", "h_dist_top3", "h_course_runs", "h_course_top3",
    "h_heavy_runs", "h_heavy_top3", "h_jockey_runs", "h_jockey_top3",
    # 人
    "j_n", "j_win", "j_top3", "j365_n", "j365_win", "j365_top3",
    "t_n", "t_win", "t_top3", "t365_n", "t365_win", "t365_top3",
] + [f"{c}_{s}" for c in RELATIVE_COLS for s in ("rrank", "rdiff")]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = prepare(raw)
    df = add_horse_history(df)
    df = add_people_stats(df)
    df = add_race_features(df)
    df = add_relative_features(df)
    df = df.sort_values(["date", "race_id", "umaban"]).reset_index(drop=True)
    cols = list(dict.fromkeys(ID_COLS + ["has_result", "n_runners"] + FEATURES))
    return df[cols]


def relevance_label(finish_pos: pd.Series) -> pd.Series:
    """ランキング学習用ラベル: 1着=3, 2着=2, 3着=1, それ以外=0."""
    return (4 - finish_pos).clip(lower=0, upper=3).fillna(0).astype(int)
