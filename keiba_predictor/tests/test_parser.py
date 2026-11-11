from keiba.jvdata import parser, spec
from keiba.jvlink_client import to_sjis
from keiba.synthetic import make_record

KEY = dict(year="2025", monthday="1116", jyo="05", kai=5, nichi=4, race_no=11)


def test_se_roundtrip():
    line = make_record(spec.SE, spec.SE_LENGTH, {
        "record_type": "SE", "data_kubun": "7", "made_date": "20251117", **KEY,
        "wakuban": 3, "umaban": 5, "horse_id": "2021104567", "horse_name": "テストホース",
        "sex": "1", "age": 4, "trainer_id": "01234", "jockey_id": "05678", "jockey_name": "騎手太郎",
        "futan": 575, "horse_weight": "482", "weight_sign": "-", "weight_diff": "006", "ijou": "0",
        "kakutei_jyuni": 2, "time": "1345", "odds": "0123", "ninki": 4, "haron_last3": "338",
        "time_diff": "+002", "corner4": 7,
    })
    assert len(line) == spec.SE_LENGTH
    rec = parser.parse_se(line.rstrip(b"\r\n"))
    assert rec["race_id"] == "2025111605050411"
    assert rec["horse_name"] == "テストホース"
    assert rec["jockey_name"] == "騎手太郎"
    assert rec["futan"] == 57.5
    assert (rec["horse_weight"], rec["weight_diff"]) == (482, -6)
    assert rec["finish_pos"] == 2
    assert rec["time"] == 94.5
    assert rec["odds"] == 12.3
    assert rec["last3f"] == 33.8
    assert rec["time_diff"] == 0.2
    assert rec["corner4"] == 7


def test_ra_roundtrip():
    line = make_record(spec.RA, spec.RA_LENGTH, {
        "record_type": "RA", "data_kubun": "2", "made_date": "20251114", **KEY,
        "race_name": "ジャパンカップ", "grade": "A", "distance": 2400, "track_code": "11",
        "syusso_tosu": 18, "turf_going": "1", "dirt_going": "0", "jyoken_young": "999",
    })
    assert len(line) == spec.RA_LENGTH
    rec = parser.parse_ra(line.rstrip(b"\r\n"))
    assert rec["date"] == "2025-11-16"
    assert rec["race_name"] == "ジャパンカップ"
    assert rec["distance"] == 2400
    assert rec["grade"] == "A"
    assert rec["syusso_tosu"] == 18


def test_value_parsers():
    assert parser.parse_race_time("0000") is None
    assert parser.parse_race_time("2012") == 121.2
    assert parser.parse_weight("999", " ", "   ") == (None, None)  # 計量不能
    assert parser.parse_weight("460", "+", "010") == (460, 10)
    assert parser.parse_signed_tenths("-003") == -0.3


def test_sjis_replacement_keeps_byte_length():
    # cp932 に無い文字 (例: 「𠮷」) も 2 バイト扱いで置換され, 固定長がずれない
    assert len(to_sjis("𠮷野")) == 4
    assert to_sjis("ABC") == b"ABC"


def test_wrong_length_records_are_skipped(tmp_path):
    p = tmp_path / "x.jv"
    p.write_bytes(b"SE7short\r\n")
    assert list(parser.iter_records(p)) == []
