import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def synthetic_db(tmp_path_factory):
    from keiba import db
    from keiba.synthetic import write_synthetic

    d = tmp_path_factory.mktemp("data")
    _, _, target = write_synthetic(d / "raw", seed=1, weeks=30, races_per_day=8, n_horses=900)
    conn = db.connect(d / "keiba.sqlite")
    db.load_raw_dir(conn, d / "raw")
    return conn, target


@pytest.fixture(scope="session")
def raw_df(synthetic_db):
    from keiba import db

    return db.read_dataset(synthetic_db[0])


@pytest.fixture(scope="session")
def feat(raw_df):
    from keiba.features import build_features

    return build_features(raw_df)
