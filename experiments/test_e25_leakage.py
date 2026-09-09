"""Leakage / causality unit tests for the E25 floor, encodings, and congestion.

Run:  python -m pytest experiments/test_e25_leakage.py -q
These use synthetic frames only — no training parquet.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e25_features import (  # noqa: E402
    CAT_COLS,
    CAT_UNMATCHED,
    NUM_COLS,
    NUM_UNMATCHED,
    add_calendar,
    add_congestion,
    assert_ranking_safe_features,
    attach_encodings_oof,
    attach_floors_time_forward,
    back_transform,
    log_excess,
    smear_factor,
    _p10_tables,
)


def _ts(h, m=0, day=1, month=3):
    return datetime(2025, month, day, h, m, tzinfo=timezone.utc)


def _dep_frame(rows: list[dict]) -> pl.DataFrame:
    n = len(rows)
    base = {
        "MVT_ID_mvt": list(range(n)),
        "airport": ["LSZH"] * n,
        "STAND_mvt": ["A1"] * n,
        "RUNWAY_mvt": ["28"] * n,
        "AIRCRAFT_OPERATOR_flt": ["OP1"] * n,
        "AIRCRAFT_TYPE_mvt": ["A320"] * n,
        "WK_TBL_CAT_flt": ["M"] * n,
        "MARKET_SEGMENT_flt": ["Mainline"] * n,
        "FLIGHT_TYPE_flt": ["S"] * n,
        "ADES_mvt": ["LFPG"] * n,
        "unmatched": [False] * n,
        "month": [3] * n,
        "y": [600.0] * n,
        "AOBT_3_flt": [_ts(10)] * n,
        "MVT_TIME_UTC_mvt": [_ts(10, 15)] * n,
        "SCHED_TIME_UTC_mvt": [_ts(10)] * n,
        "EOBT_1_flt": [_ts(10)] * n,
        "LOBT_flt": [_ts(10)] * n,
        "IOBT_flt": [_ts(10)] * n,
    }
    for i, r in enumerate(rows):
        for k, v in r.items():
            col = base[k]
            col[i] = v
    return pl.DataFrame(base)


def test_feature_lists_exclude_block_and_taxitime():
    assert_ranking_safe_features(NUM_COLS + CAT_COLS + NUM_UNMATCHED + CAT_UNMATCHED)
    blob = " ".join(NUM_COLS + CAT_COLS + NUM_UNMATCHED + CAT_UNMATCHED)
    assert "BLOCK" not in blob
    assert "TAXITIME" not in blob


def test_encoding_excludes_own_row_target():
    """Four rows, same stand×runway, four different months: OOF mean ≠ in-sample mean."""
    rows = []
    ys = [400.0, 800.0, 1200.0, 1600.0]
    for i, (y, month) in enumerate(zip(ys, [2, 3, 4, 5])):
        rows.append(
            {
                "y": y,
                "month": month,
                "MVT_TIME_UTC_mvt": _ts(10, 15, day=1, month=month),
                "AOBT_3_flt": _ts(10, 0, day=1, month=month),
            }
        )
    df = add_calendar(_dep_frame(rows))
    tr, va = attach_encodings_oof(df, df.head(0))
    enc = tr.sort("month")["enc_stand_rwy"].to_numpy()
    in_sample = float(np.mean(ys))
    # Each row's encoding is JS-shrink of the other three toward the other-three airport mean.
    # With n=3, k=20, it is pulled hard toward the prior, but it must NOT equal the 4-row mean
    # and must not equal its own y.
    for i, y in enumerate(ys):
        assert enc[i] != pytest.approx(y, abs=1e-6)
        assert enc[i] != pytest.approx(in_sample, abs=1e-3)
        others = [ys[j] for j in range(4) if j != i]
        oof_mean = float(np.mean(others))
        # JS: (3*oof_mean + 20*oof_mean) / 23 == oof_mean, because prior is also the other-three mean
        # (one airport, three rows). So encoding == mean of the other rows, not including self.
        assert enc[i] == pytest.approx(oof_mean, abs=1e-4)


def test_encoding_val_does_not_use_val_y():
    train = add_calendar(
        _dep_frame(
            [
                {"y": 500.0, "month": 2, "MVT_TIME_UTC_mvt": _ts(10, 15, month=2), "AOBT_3_flt": _ts(10, month=2)},
                {"y": 700.0, "month": 3, "MVT_TIME_UTC_mvt": _ts(10, 15, month=3), "AOBT_3_flt": _ts(10, month=3)},
            ]
        )
    )
    val = add_calendar(
        _dep_frame(
            [
                {
                    "y": 50_000.0,  # poison if leaked
                    "month": 7,
                    "MVT_TIME_UTC_mvt": _ts(10, 15, month=7),
                    "AOBT_3_flt": _ts(10, month=7),
                }
            ]
        )
    )
    # IDs must be unique across concat-style usage; rebuild val with new id via join path
    val = val.with_columns(pl.lit(99).alias("MVT_ID_mvt"))
    _, va = attach_encodings_oof(train, val)
    enc = float(va["enc_stand_rwy"][0])
    train_mean = 600.0
    assert enc == pytest.approx(train_mean, abs=1e-4)
    assert enc < 1000.0  # poison 50k did not enter


def test_floor_time_forward_excludes_future_and_self():
    """Three months, one row each, same triple. Floor for month 4 may use month 3 y; not vice versa."""
    rows = []
    for month, y in [(3, 400.0), (4, 900.0), (5, 1500.0)]:
        rows.append(
            {
                "y": y,
                "month": month,
                "MVT_TIME_UTC_mvt": _ts(10, 15, month=month),
                "AOBT_3_flt": _ts(10, month=month),
            }
        )
    train = add_calendar(_dep_frame(rows))
    val = train.head(0)
    tr, _ = attach_floors_time_forward(train, val)
    tr = tr.sort("month")
    floors = tr["floor"].to_numpy()
    # First train month has no earlier history → LOMO uses later months. That is
    # allowed (no self-row). Months 4 and 5 must be strictly expanding.
    # Month 4 history = month 3 only (n=1). Hierarchical p10 of a singleton is 400,
    # shrunk toward airport p10 which is also 400. Floor ≈ 400.
    assert floors[1] == pytest.approx(400.0, abs=1.0)
    # Month 5 history = months 3+4, y={400,900}. p10 of two points is near 400.
    # Must not equal 1500 (self) and must be well below 1500.
    assert floors[2] < 1000.0
    assert floors[2] != pytest.approx(1500.0, abs=1.0)


def test_floor_tables_exclude_unmatched_extremes():
    rows = [{"y": 600.0, "unmatched": False, "month": 3}] * 20
    rows.append({"y": 80_000.0, "unmatched": True, "month": 3, "MVT_ID_mvt": 99})
    # _dep_frame overwrites by index; build explicitly
    n = 21
    df = add_calendar(
        pl.DataFrame(
            {
                "MVT_ID_mvt": list(range(n)),
                "airport": ["LSZH"] * n,
                "STAND_mvt": ["A1"] * n,
                "RUNWAY_mvt": ["28"] * n,
                "AIRCRAFT_OPERATOR_flt": ["OP1"] * n,
                "AIRCRAFT_TYPE_mvt": ["A320"] * n,
                "WK_TBL_CAT_flt": ["M"] * n,
                "MARKET_SEGMENT_flt": ["Mainline"] * n,
                "FLIGHT_TYPE_flt": ["S"] * n,
                "ADES_mvt": ["LFPG"] * n,
                "unmatched": [False] * 20 + [True],
                "month": [3] * n,
                "y": [600.0] * 20 + [80_000.0],
                "AOBT_3_flt": [_ts(10)] * n,
                "MVT_TIME_UTC_mvt": [_ts(10, 15)] * n,
                "SCHED_TIME_UTC_mvt": [_ts(10)] * n,
                "EOBT_1_flt": [_ts(10)] * n,
                "LOBT_flt": [_ts(10)] * n,
                "IOBT_flt": [_ts(10)] * n,
            }
        )
    )
    tabs = _p10_tables(df)
    ap_p10 = float(tabs["ap"]["p10"][0])
    assert ap_p10 == pytest.approx(600.0, abs=1.0)


def test_congestion_self_excluded_and_window_and_runway():
    t0 = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    # 0: query. 1: same rwy AOBT 5 min earlier. 2: same rwy 20 min earlier.
    # 3: other rwy 5 min earlier. 4: same rwy AOBT 5 min later (future, not counted).
    rows = [
        {"AOBT_3_flt": t0, "MVT_TIME_UTC_mvt": t0 + timedelta(minutes=15), "RUNWAY_mvt": "28"},
        {
            "AOBT_3_flt": t0 - timedelta(minutes=5),
            "MVT_TIME_UTC_mvt": t0 + timedelta(minutes=10),
            "RUNWAY_mvt": "28",
        },
        {
            "AOBT_3_flt": t0 - timedelta(minutes=20),
            "MVT_TIME_UTC_mvt": t0 - timedelta(minutes=5),
            "RUNWAY_mvt": "28",
        },
        {
            "AOBT_3_flt": t0 - timedelta(minutes=5),
            "MVT_TIME_UTC_mvt": t0 + timedelta(minutes=8),
            "RUNWAY_mvt": "10",
        },
        {
            "AOBT_3_flt": t0 + timedelta(minutes=5),
            "MVT_TIME_UTC_mvt": t0 + timedelta(minutes=20),
            "RUNWAY_mvt": "28",
        },
    ]
    dep = add_calendar(_dep_frame(rows))
    arr = pl.DataFrame(
        {
            "airport": ["LSZH", "LSZH", "LSZH"],
            "RUNWAY_mvt": ["28", "10", "28"],
            "MVT_TIME_UTC_mvt": [
                t0 - timedelta(minutes=4),  # same rwy landing
                t0 - timedelta(minutes=3),  # other rwy landing (crossing proxy)
                t0 + timedelta(minutes=2),  # after query AOBT — excluded
            ],
        }
    )
    out = add_congestion(dep, arr).sort("MVT_ID_mvt")
    q = out.row(0, named=True)
    assert q["dep_rwy_aobt_15m"] == pytest.approx(1.0)  # only row 1
    assert q["dep_rwy_aobt_10m"] == pytest.approx(1.0)
    assert q["arr_rwy_15m"] == pytest.approx(1.0)
    assert q["arr_other_rwy_15m"] == pytest.approx(1.0)
    # row 1 (5 min earlier) should not count the query (future) as a trailing AOBT
    r1 = out.row(1, named=True)
    assert r1["dep_rwy_aobt_15m"] == pytest.approx(1.0)  # the 20-min-earlier flight is 15 min before row1? 20-5=15
    # t0-20 is exactly 15 min before t0-5, included in [q-15m, q]
    # row 2 (20 min earlier) should see zero trailing same-rwy AOBT
    r2 = out.row(2, named=True)
    assert r2["dep_rwy_aobt_15m"] == pytest.approx(0.0)


def test_congestion_does_not_use_taxitime_or_block():
    t0 = datetime(2025, 3, 1, 12, 0, tzinfo=timezone.utc)
    dep = add_calendar(
        _dep_frame(
            [
                {"AOBT_3_flt": t0, "MVT_TIME_UTC_mvt": t0 + timedelta(minutes=12), "y": 99999.0},
                {
                    "AOBT_3_flt": t0 - timedelta(minutes=3),
                    "MVT_TIME_UTC_mvt": t0 + timedelta(minutes=9),
                    "y": 12.0,
                },
            ]
        )
    )
    arr = pl.DataFrame(
        {
            "airport": ["EDDF"],
            "RUNWAY_mvt": ["07C"],
            "MVT_TIME_UTC_mvt": [datetime(2025, 3, 1, 1, 0, tzinfo=timezone.utc)],
        }
    )
    out = add_congestion(dep, arr)
    # Counts depend only on AOBT timestamps, not on y.
    assert out["dep_rwy_aobt_15m"][0] == pytest.approx(1.0)


def test_smearing_corrects_naive_exp_bias():
    rng = np.random.default_rng(0)
    z_true = rng.normal(5.0, 0.4, size=5000)
    z_pred = np.full(5000, 5.0)  # perfect mean in log space
    floor = np.full(5000, 400.0)
    smear = smear_factor(z_true, z_pred)
    naive = naive_mean = float(np.mean(floor + np.exp(z_pred)))
    duan = float(np.mean(back_transform(z_pred, floor, smear)))
    actual = float(np.mean(floor + np.exp(z_true)))
    assert smear > 1.0
    assert abs(duan - actual) < abs(naive_mean - actual)


def test_log_excess_clips_below_floor():
    z = log_excess(np.array([100.0, 800.0]), np.array([400.0, 400.0]))
    assert z[0] == pytest.approx(np.log(1.0))
    assert z[1] == pytest.approx(np.log(400.0))


def test_val_floor_ignores_val_y():
    train = add_calendar(
        _dep_frame(
            [{"y": 500.0, "month": 2, "MVT_TIME_UTC_mvt": _ts(10, 15, month=2), "AOBT_3_flt": _ts(10, month=2)}]
        )
    )
    val = add_calendar(
        _dep_frame(
            [
                {
                    "y": 90_000.0,
                    "month": 7,
                    "MVT_TIME_UTC_mvt": _ts(10, 15, month=7),
                    "AOBT_3_flt": _ts(10, month=7),
                }
            ]
        )
    ).with_columns(pl.lit(99).alias("MVT_ID_mvt"))
    _, va = attach_floors_time_forward(train, val)
    assert float(va["floor"][0]) == pytest.approx(500.0, abs=1.0)
