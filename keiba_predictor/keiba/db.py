"""パース済みレコードを SQLite に蓄積する.

同じレース/馬について出馬表 -> 速報成績 -> 確定成績と何度もレコードが届くので,
(データ作成日, データ区分) が新しいものだけで上書きする。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from .jvdata import parser

log = logging.getLogger(__name__)

# データ区分の新しさ (大きいほど後). '0' は削除.
_KUBUN_RANK = {"1": "1", "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "A": "7", "B": "7", "9": "8"}

RACE_COLS = [
    "race_id", "data_kubun", "made_date", "date", "jyo", "kai", "nichi", "race_no", "race_name",
    "grade", "syubetsu", "kigo", "juryo", "jyoken", "distance", "track_code", "course_kubun",
    "prize1", "start_time", "toroku_tosu", "syusso_tosu", "weather", "turf_going", "dirt_going",
    "race_first3f", "race_last3f",
]
ENTRY_COLS = [
    "race_id", "umaban", "data_kubun", "made_date", "wakuban", "horse_id", "horse_name", "sex", "age",
    "tozai", "trainer_id", "trainer_name", "owner_id", "jockey_id", "jockey_name", "minarai", "futan",
    "blinker", "horse_weight", "weight_diff", "ijou", "finish_pos", "time", "time_diff",
    "corner1", "corner2", "corner3", "corner4", "odds", "ninki", "prize", "last3f", "last4f", "kyakusitu",
]
WEIGHT_COLS = ["race_id", "umaban", "announced", "horse_weight", "weight_diff"]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS races ({", ".join(RACE_COLS)}, version TEXT, PRIMARY KEY (race_id));
CREATE TABLE IF NOT EXISTS entries ({", ".join(ENTRY_COLS)}, version TEXT, PRIMARY KEY (race_id, umaban));
CREATE TABLE IF NOT EXISTS weights ({", ".join(WEIGHT_COLS)}, PRIMARY KEY (race_id, umaban));
CREATE TABLE IF NOT EXISTS loaded_files (name TEXT PRIMARY KEY, records INTEGER, loaded_at TEXT);
CREATE INDEX IF NOT EXISTS idx_entries_horse ON entries (horse_id);
CREATE INDEX IF NOT EXISTS idx_races_date ON races (date);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    return conn


def _upsert(conn: sqlite3.Connection, table: str, cols: list[str], keys: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    allcols = cols + ["version"]
    placeholders = ", ".join("?" for _ in allcols)
    updates = ", ".join(f"{c} = excluded.{c}" for c in allcols if c not in keys)
    sql = (
        f"INSERT INTO {table} ({', '.join(allcols)}) VALUES ({placeholders}) "
        f"ON CONFLICT ({', '.join(keys)}) DO UPDATE SET {updates} "
        f"WHERE excluded.version >= {table}.version"
    )
    conn.executemany(sql, [[r.get(c) for c in cols] + [r["version"]] for r in rows])


def _split_versioned(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    keep, delete = [], []
    for r in rows:
        if r["data_kubun"] == "0":
            delete.append(r)
        else:
            r["version"] = r["made_date"] + _KUBUN_RANK.get(r["data_kubun"], "0")
            keep.append(r)
    return keep, delete


def store_records(conn: sqlite3.Connection, races: list[dict], entries: list[dict], weights: list[dict]) -> None:
    races_keep, races_del = _split_versioned(races)
    entries_keep, entries_del = _split_versioned(entries)
    _upsert(conn, "races", RACE_COLS, ["race_id"], races_keep)
    _upsert(conn, "entries", ENTRY_COLS, ["race_id", "umaban"], entries_keep)
    conn.executemany("DELETE FROM races WHERE race_id = ?", [(r["race_id"],) for r in races_del])
    conn.executemany(
        "DELETE FROM entries WHERE race_id = ? AND umaban = ?", [(r["race_id"], r["umaban"]) for r in entries_del]
    )
    if weights:
        conn.executemany(
            f"INSERT INTO weights ({', '.join(WEIGHT_COLS)}) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (race_id, umaban) DO UPDATE SET announced = excluded.announced, "
            "horse_weight = excluded.horse_weight, weight_diff = excluded.weight_diff "
            "WHERE excluded.announced >= weights.announced",
            [[w[c] for c in WEIGHT_COLS] for w in weights],
        )


def load_raw_file(conn: sqlite3.Connection, path: Path) -> int:
    races, entries, weights = [], [], []
    for rtype, rec in parser.iter_records(path):
        if rtype == "RA":
            races.append(rec)
        elif rtype == "SE":
            entries.append(rec)
        elif rtype == "WH":
            weights.extend(rec)
    with conn:
        store_records(conn, races, entries, weights)
        n = len(races) + len(entries) + len(weights)
        conn.execute(
            "INSERT OR REPLACE INTO loaded_files VALUES (?, ?, datetime('now'))", (path.name, n)
        )
    return n


def load_raw_dir(conn: sqlite3.Connection, raw_dir: Path, force: bool = False) -> int:
    """raw_dir 内の未取込ファイルを古い順に取り込む."""
    done = {r[0] for r in conn.execute("SELECT name FROM loaded_files")}
    total = 0
    for path in sorted(raw_dir.glob("*.jv")):
        if path.name in done and not force:
            continue
        n = load_raw_file(conn, path)
        log.info("loaded %s (%d records)", path.name, n)
        total += n
    return total


def read_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    """出走馬 1 頭 = 1 行のデータフレーム (レース情報結合済み)."""
    races = pd.read_sql_query(f"SELECT {', '.join(RACE_COLS)} FROM races", conn)
    entries = pd.read_sql_query(f"SELECT {', '.join(ENTRY_COLS)} FROM entries", conn)
    weights = pd.read_sql_query("SELECT race_id, umaban, horse_weight AS wh_weight, weight_diff AS wh_diff FROM weights", conn)
    races = races.rename(columns={"data_kubun": "race_kubun", "made_date": "race_made_date"})
    df = entries.merge(races, on="race_id", how="inner")
    if not weights.empty:
        df = df.merge(weights, on=["race_id", "umaban"], how="left")
        df["horse_weight"] = df["horse_weight"].fillna(df["wh_weight"])
        df["weight_diff"] = df["weight_diff"].fillna(df["wh_diff"])
        df = df.drop(columns=["wh_weight", "wh_diff"])
    return df
