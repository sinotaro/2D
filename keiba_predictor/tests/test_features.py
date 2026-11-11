import numpy as np
import pandas as pd

from keiba.features import FEATURES, build_features


def test_no_future_leakage(raw_df, feat):
    """ある日の結果を書き換えても, その日以前の特徴量は一切変わらないこと."""
    dates = sorted(pd.to_datetime(raw_df["date"]).unique())
    cut = dates[len(dates) // 2]
    tampered = raw_df.copy()
    on_or_after = pd.to_datetime(tampered["date"]) >= cut
    rng = np.random.default_rng(0)
    tampered.loc[on_or_after, "finish_pos"] = rng.permutation(tampered.loc[on_or_after, "finish_pos"].to_numpy())
    tampered.loc[on_or_after, "time"] = tampered.loc[on_or_after, "time"] + rng.normal(0, 2, on_or_after.sum())
    tampered.loc[on_or_after, "last3f"] = 40.0
    tampered.loc[on_or_after, "corner4"] = 1

    feat2 = build_features(tampered)
    key = ["race_id", "umaban"]
    a = feat[feat["date"] <= cut].set_index(key, drop=False).sort_index()[FEATURES]
    b = feat2[feat2["date"] <= cut].set_index(key, drop=False).sort_index()[FEATURES]
    pd.testing.assert_frame_equal(a, b)
    # 当然, 翌日以降は変わる (= 過去成績を実際に使っている)
    a2 = feat[feat["date"] > cut].set_index(key, drop=False).sort_index()[FEATURES]
    b2 = feat2[feat2["date"] > cut].set_index(key, drop=False).sort_index()[FEATURES]
    assert not a2.equals(b2)


def test_upcoming_races_get_history_features(feat, synthetic_db):
    _, target = synthetic_db
    up = feat[feat["date"] == pd.Timestamp(target)]
    assert len(up) > 0
    assert not up["has_result"].any()
    assert up["h_runs"].gt(0).mean() > 0.9
    assert up["j_n"].gt(0).all()


def test_jockey_stats_exclude_same_day(feat):
    # 同じ騎手・同じ日の行は全て同じ値 (当日の他レース結果を使っていない)
    sub = feat.dropna(subset=["j_win"])
    nunique = sub.groupby([sub["jockey_name"], sub["date"]])["j_win"].nunique()
    assert (nunique == 1).all()
