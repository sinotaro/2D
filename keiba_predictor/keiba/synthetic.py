"""動作確認用: JV-Data と同じ固定長フォーマットの架空レースデータを生成する.

JV-Link が無い環境 (Mac/Linux, CI) でもパイプライン全体を試せるようにするためのもの。
馬の能力・距離適性・芝ダート適性・騎手の腕などの潜在変数から結果を生成するので,
過去成績から学習したモデルがランダムより当たるかを確認できる。
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .jvdata import spec

KANA = list("アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワン")


def _put(buf: bytearray, field: spec.Field, value, offset: int = 0) -> None:
    s = field.slice
    if value is None:
        return
    if field.kind == "int":
        data = str(int(value)).zfill(field.length).encode("ascii")
    elif field.kind == "raw":
        data = str(value).ljust(field.length).encode("cp932")
    else:
        data = str(value).encode("cp932")
        data = data[: field.length]
        pad = field.length - len(data)
        data += "　".encode("cp932") * (pad // 2) + b" " * (pad % 2)
    if len(data) != field.length:
        raise ValueError(f"{field.name}: {value!r} does not fit in {field.length} bytes")
    buf[s.start + offset : s.stop + offset] = data


def make_record(fields: list[spec.Field], length: int, values: dict) -> bytes:
    buf = bytearray(b" " * (length - 2))
    by_name = {f.name: f for f in fields}
    for name, value in values.items():
        _put(buf, by_name[name], value)
    return bytes(buf) + b"\r\n"


def _fmt_time(sec: float) -> str:
    t = int(round(sec * 10))
    m, rest = divmod(t, 600)
    return f"{m}{rest // 10:02d}{rest % 10}"


class Simulator:
    def __init__(self, seed: int = 0, n_horses: int = 1500, n_jockeys: int = 60, n_trainers: int = 50):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        self.n_horses = n_horses
        self.ability = r.normal(0, 1, n_horses)
        self.turf_apt = r.normal(0, 0.6, n_horses)
        self.opt_dist = r.choice([1200, 1400, 1600, 1800, 2000, 2400], n_horses)
        self.style = r.uniform(0, 1, n_horses)  # 0=逃げ 1=追込
        self.sex = r.choice(["1", "2", "3"], n_horses, p=[0.5, 0.45, 0.05])
        self.birth_year = r.integers(0, 4, n_horses)
        self.base_weight = r.normal(475, 25, n_horses).round()
        self.names = ["".join(r.choice(KANA, r.integers(3, 9))) for _ in range(n_horses)]
        self.jockey_skill = r.normal(0, 0.35, n_jockeys)
        self.trainer_skill = r.normal(0, 0.25, n_trainers)
        self.horse_trainer = r.integers(0, n_trainers, n_horses)
        self.last_run = np.full(n_horses, -999)
        self.weight = self.base_weight.copy()

    def _race(self, day_idx: int, d: date, jyo: str, race_no: int, kai: int, nichi: int, with_result: bool):
        r = self.rng
        turf = r.random() < 0.55
        distance = int(r.choice([1200, 1400, 1600, 1800, 2000, 2200, 2400] if turf else [1200, 1400, 1700, 1800, 2100]))
        going = int(r.choice([1, 2, 3, 4], p=[0.7, 0.15, 0.1, 0.05]))
        rested = np.flatnonzero(day_idx - self.last_run >= 14)
        n = int(min(r.integers(8, 17), len(rested)))
        horses = r.choice(rested, n, replace=False)
        jockeys = r.choice(len(self.jockey_skill), n, replace=False)
        futan = r.choice([54.0, 55.0, 56.0, 57.0, 58.0], n)

        perf = (
            self.ability[horses]
            + np.where(turf, self.turf_apt[horses], -self.turf_apt[horses])
            - np.abs(distance - self.opt_dist[horses]) / 500
            + self.jockey_skill[jockeys]
            + self.trainer_skill[self.horse_trainer[horses]]
            - (futan - 55) * 0.05
            - np.where(going >= 3, self.style[horses] * 0.3, 0)
        )
        noisy = perf + r.normal(0, 0.9, n)
        order = np.argsort(-noisy)
        finish = np.empty(n, int)
        finish[order] = np.arange(1, n + 1)
        market = perf + r.normal(0, 0.5, n)
        p = np.exp(market * 1.5) / np.exp(market * 1.5).sum()
        odds = np.clip(np.round(0.8 / p, 1), 1.1, 999.9)
        ninki = np.empty(n, int)
        ninki[np.argsort(-p)] = np.arange(1, n + 1)
        base = distance / 1000 * 60 * (1.0 if turf else 1.04) + (going - 1) * 0.6
        times = base + (noisy.max() - noisy) * 0.35
        last3f = 34.5 + (1 if not turf else 0) - (noisy - noisy.mean()) * 0.4 + r.normal(0, 0.3, n)
        corner4 = np.empty(n, int)
        corner4[np.argsort(self.style[horses] + r.normal(0, 0.2, n))] = np.arange(1, n + 1)
        prize = int(r.choice([5000, 7000, 10000, 15000, 20000, 40000]))

        key = dict(year=f"{d.year}", monthday=f"{d.month:02d}{d.day:02d}", jyo=jyo, kai=kai, nichi=nichi, race_no=race_no)
        kubun = "7" if with_result else "2"
        made = d.strftime("%Y%m%d") if not with_result else (d + timedelta(days=1)).strftime("%Y%m%d")
        ra = make_record(spec.RA, spec.RA_LENGTH, {
            "record_type": "RA", "data_kubun": kubun, "made_date": made, **key,
            "race_name": f"架空ステークス{race_no}", "race_name_short": f"架空{race_no}",
            "grade": " ", "syubetsu": "13", "kigo": "000", "juryo": "4",
            "jyoken_young": str(r.choice(["703", "005", "010", "016", "999"])),
            "distance": distance, "track_code": "11" if turf else "23", "course_kubun": "  ",
            "honsyokin1": prize, "start_time": f"{10 + race_no // 2:02d}{(race_no % 2) * 30:02d}",
            "toroku_tosu": n, "syusso_tosu": n, "nyusen_tosu": n if with_result else 0,
            "weather": "1", "turf_going": str(going) if turf else "0", "dirt_going": "0" if turf else str(going),
        })
        out = [ra]
        sort_idx = np.arange(n)
        for i in sort_idx:
            h = horses[i]
            umaban = i + 1
            new_w = self.weight[h] + r.normal(0, 4)
            diff = int(round(new_w - self.weight[h]))
            vals = {
                "record_type": "SE", "data_kubun": kubun, "made_date": made, **key,
                "wakuban": min(8, (i * 8) // n + 1), "umaban": umaban,
                "horse_id": f"{2018 + self.birth_year[h]}1{h:05d}", "horse_name": self.names[h],
                "sex": self.sex[h], "hinsyu": "1", "keiro": "01",
                "age": d.year - (2018 + self.birth_year[h]), "tozai": "1" if h % 2 else "2",
                "trainer_id": f"{self.horse_trainer[h]:05d}", "trainer_name": f"調教師{self.horse_trainer[h]}",
                "owner_id": "000001", "futan": int(futan[i] * 10), "blinker": "0",
                "jockey_id": f"{jockeys[i]:05d}", "jockey_name": f"騎手{jockeys[i]}", "minarai": "0",
                "horse_weight": f"{int(round(new_w)):03d}", "weight_sign": "-" if diff < 0 else "+",
                "weight_diff": f"{abs(diff):03d}", "ijou": "0",
            }
            if with_result:
                vals.update({
                    "nyusen_jyuni": finish[i], "kakutei_jyuni": finish[i], "time": _fmt_time(times[i]),
                    "corner4": corner4[i], "odds": f"{int(odds[i] * 10):04d}", "ninki": ninki[i],
                    "haron_last3": f"{int(round(last3f[i] * 10)):03d}",
                    "time_diff": f"{'+' if finish[i] > 1 else '-'}{int(round(abs(times[i] - np.sort(times)[1 if finish[i] == 1 else 0]) * 10)):03d}",
                })
                self.weight[h] = new_w
            self.last_run[h] = day_idx
            out.append(make_record(spec.SE, spec.SE_LENGTH, vals))
        return out

    def generate(self, start: date, weeks: int, races_per_day: int = 12, venues=("05", "08")):
        """過去 weeks 週分の成績と, 最終週の出馬表 (結果なし) を返す."""
        history, upcoming = [], []
        day_idx = 0
        for w in range(weeks + 1):
            for dow in (5, 6):  # 土日
                d = start + timedelta(weeks=w, days=dow - start.weekday())
                day_idx = (d - start).days
                for v_i, jyo in enumerate(venues):
                    for rn in range(1, races_per_day + 1):
                        recs = self._race(day_idx, d, jyo, rn, kai=(w // 4) % 5 + 1, nichi=(w % 4) * 2 + dow - 4,
                                          with_result=w < weeks)
                        (history if w < weeks else upcoming).extend(recs)
            # 少しずつ能力が変化 (成長・衰え)
            self.ability += self.rng.normal(0, 0.03, self.n_horses)
        return history, upcoming


def write_synthetic(raw_dir: Path, seed: int = 0, weeks: int = 80, start: date = date(2024, 1, 6),
                    races_per_day: int = 12, n_horses: int = 1500) -> tuple[Path, Path, str]:
    """生データファイルを書き出し, (成績ファイル, 出馬表ファイル, 予想対象日) を返す."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    sim = Simulator(seed=seed, n_horses=n_horses)
    history, upcoming = sim.generate(start, weeks, races_per_day=races_per_day)
    hist_path = raw_dir / "SYN_history.jv"
    week_path = raw_dir / "SYN_week.jv"
    hist_path.write_bytes(b"".join(history))
    week_path.write_bytes(b"".join(upcoming))
    target = start + timedelta(weeks=weeks, days=5 - start.weekday())
    return hist_path, week_path, target.isoformat()
