"""JV-Data 固定長レコードのフィールド定義.

位置は JV-Data 仕様書 (Ver.4.x) に基づく 1 始まりのバイト位置 (Shift-JIS)。
本ツールで使う項目のみを定義している。仕様書の改版で位置が変わった場合は
ここだけを直せばよい。

型:
  "str"   文字列 (前後の半角/全角スペースを除去)
  "int"   整数 (空白・非数値は None)
  "raw"   文字列 (除去しない。コード値用)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Field:
    name: str
    start: int  # 1 始まり
    length: int
    kind: str = "str"

    @property
    def slice(self) -> slice:
        return slice(self.start - 1, self.start - 1 + self.length)


# 全レコード共通ヘッダ + レース識別キー
_HEADER = [
    Field("record_type", 1, 2, "raw"),
    Field("data_kubun", 3, 1, "raw"),
    Field("made_date", 4, 8, "raw"),
]
_RACE_KEY = [
    Field("year", 12, 4, "raw"),
    Field("monthday", 16, 4, "raw"),
    Field("jyo", 20, 2, "raw"),
    Field("kai", 22, 2, "int"),
    Field("nichi", 24, 2, "int"),
    Field("race_no", 26, 2, "int"),
]

# RA: レース詳細 (1272 byte)
RA = _HEADER + _RACE_KEY + [
    Field("weekday", 28, 1, "raw"),
    Field("race_name", 33, 60),
    Field("race_name_sub", 93, 60),
    Field("race_name_short", 573, 20),
    Field("grade", 615, 1, "raw"),
    Field("syubetsu", 617, 2, "raw"),  # 競走種別コード (11:2歳 12:3歳 13:3歳以上 14:4歳以上 ...)
    Field("kigo", 619, 3, "raw"),  # 競走記号コード
    Field("juryo", 622, 1, "raw"),  # 重量種別コード (1:ハンデ 2:別定 3:馬齢 4:定量)
    Field("jyoken_2", 623, 3, "raw"),
    Field("jyoken_3", 626, 3, "raw"),
    Field("jyoken_4", 629, 3, "raw"),
    Field("jyoken_5", 632, 3, "raw"),
    Field("jyoken_young", 635, 3, "raw"),  # 最若年条件 (005:1勝 010:2勝 016:3勝 701:新馬 702:未出走 703:未勝利 999:OP)
    Field("distance", 698, 4, "int"),
    Field("track_code", 706, 2, "raw"),
    Field("course_kubun", 710, 2, "raw"),
    Field("honsyokin1", 714, 8, "int"),  # 1着本賞金 (百円)
    Field("start_time", 874, 4, "raw"),
    Field("toroku_tosu", 882, 2, "int"),
    Field("syusso_tosu", 884, 2, "int"),
    Field("nyusen_tosu", 886, 2, "int"),
    Field("weather", 888, 1, "raw"),
    Field("turf_going", 889, 1, "raw"),
    Field("dirt_going", 890, 1, "raw"),
    Field("haron_first3", 970, 3, "int"),
    Field("haron_last3", 976, 3, "int"),
]
RA_LENGTH = 1272

# SE: 馬毎レース情報 (555 byte)
SE = _HEADER + _RACE_KEY + [
    Field("wakuban", 28, 1, "int"),
    Field("umaban", 29, 2, "int"),
    Field("horse_id", 31, 10, "raw"),
    Field("horse_name", 41, 36),
    Field("sex", 79, 1, "raw"),  # 1:牡 2:牝 3:セン
    Field("hinsyu", 80, 1, "raw"),
    Field("keiro", 81, 2, "raw"),
    Field("age", 83, 2, "int"),
    Field("tozai", 85, 1, "raw"),  # 1:美浦 2:栗東 3:地方 4:外国
    Field("trainer_id", 86, 5, "raw"),
    Field("trainer_name", 91, 8),
    Field("owner_id", 99, 6, "raw"),
    Field("futan", 289, 3, "int"),  # 負担重量 0.1kg
    Field("blinker", 295, 1, "raw"),
    Field("jockey_id", 297, 5, "raw"),
    Field("jockey_name", 307, 8),
    Field("minarai", 323, 1, "raw"),  # 騎手見習コード
    Field("horse_weight", 325, 3, "raw"),
    Field("weight_sign", 328, 1, "raw"),
    Field("weight_diff", 329, 3, "raw"),
    Field("ijou", 332, 1, "raw"),  # 異常区分 (0:なし 1:取消 2:発走除外 3:競走除外 4:中止 5:失格 6:再騎乗 7:降着)
    Field("nyusen_jyuni", 333, 2, "int"),
    Field("kakutei_jyuni", 335, 2, "int"),
    Field("time", 339, 4, "raw"),  # 走破タイム (分1桁+秒2桁+0.1秒1桁)
    Field("corner1", 352, 2, "int"),
    Field("corner2", 354, 2, "int"),
    Field("corner3", 356, 2, "int"),
    Field("corner4", 358, 2, "int"),
    Field("odds", 360, 4, "raw"),  # 単勝オッズ 0.1倍
    Field("ninki", 364, 2, "int"),
    Field("honsyokin", 366, 8, "int"),
    Field("haron_last4", 388, 3, "raw"),
    Field("haron_last3", 391, 3, "raw"),
    Field("time_diff", 532, 4, "raw"),  # タイム差 (符号+0.1秒3桁)
    Field("kyakusitu", 553, 1, "raw"),  # 今回レース脚質判定 (1:逃 2:先 3:差 4:追)
]
SE_LENGTH = 555

# WH: 馬体重 (速報。847 byte)
WH_HEADER = _HEADER + [
    Field("year", 12, 4, "raw"),
    Field("monthday", 16, 4, "raw"),
    Field("jyo", 20, 2, "raw"),
    Field("kai", 22, 2, "int"),
    Field("nichi", 24, 2, "int"),
    Field("race_no", 26, 2, "int"),
    Field("announced", 28, 8, "raw"),
]
WH_ITEM_START = 36
WH_ITEM_SIZE = 45
WH_ITEM_COUNT = 18
WH_ITEM = [  # 繰り返し項目内の相対位置 (1 始まり)
    Field("umaban", 1, 2, "int"),
    Field("horse_name", 3, 36),
    Field("horse_weight", 39, 3, "raw"),
    Field("weight_sign", 42, 1, "raw"),
    Field("weight_diff", 43, 3, "raw"),
]
WH_LENGTH = 847

RECORD_SPECS = {"RA": RA, "SE": SE}
RECORD_LENGTHS = {"RA": RA_LENGTH, "SE": SE_LENGTH, "WH": WH_LENGTH}
