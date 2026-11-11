
from keiba import model
from keiba.predict import format_race, predict_races


def test_train_and_predict(feat, synthetic_db, tmp_path):
    _, target = synthetic_db
    dates = feat["date"]
    valid_from = (dates.min() + (dates.max() - dates.min()) * 0.75).date().isoformat()
    models, report = model.train(feat, valid_from=valid_from, min_history_date="2024-02-15", num_rounds=300)
    v = report["validation"]
    # 架空データは能力差があるので, ランダムより明確に当たるはず
    assert v["top1_win_rate"] > 2.5 * v["random_win_rate"]
    assert v["ndcg@3"] > 0.5

    models.save(tmp_path / "m", meta=report)
    loaded = model.Models.load(tmp_path / "m")
    pred = predict_races(loaded, feat, date=target)
    assert len(pred) > 0
    s = pred.groupby("race_id")["win_prob"].sum()
    assert ((s - 1).abs() < 1e-6).all()
    for _, g in pred.groupby("race_id"):
        assert sorted(g["pred_rank"]) == list(range(1, len(g) + 1))
    assert "◎" in format_race(pred[pred["race_id"] == pred["race_id"].iloc[0]])
    assert pred["finish_pos"].isna().all()
