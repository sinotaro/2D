from keiba import db


def _entry(kubun, made, pos):
    return {c: None for c in db.ENTRY_COLS} | {
        "race_id": "2025111605050411", "umaban": 1, "data_kubun": kubun, "made_date": made, "finish_pos": pos,
    }


def test_newer_record_wins_and_older_is_ignored(tmp_path):
    conn = db.connect(tmp_path / "t.sqlite")
    db.store_records(conn, [], [_entry("7", "20251117", 3)], [])
    db.store_records(conn, [], [_entry("2", "20251114", None)], [])  # 古い出馬表が後から来ても上書きしない
    assert conn.execute("SELECT finish_pos FROM entries").fetchone()[0] == 3
    db.store_records(conn, [], [_entry("7", "20251120", 4)], [])  # 訂正
    assert conn.execute("SELECT finish_pos FROM entries").fetchone()[0] == 4
    db.store_records(conn, [], [_entry("0", "20251121", None)], [])  # 削除
    assert conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 0
