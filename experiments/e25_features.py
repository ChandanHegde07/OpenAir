"""E25 rebuild: ranking-safe floor, encodings, congestion, log-excess target.

Information set (post-ops reconstruction, matching ranking):
  Available: scored-row MVT/stand/runway/type/schedule/NM clocks, other DEPs'
  MVT/AOBT_3/stand/runway/NM, all ARR fields.
  Forbidden: any DEP BLOCK_TIME or TAXITIME, including other flights'.

AOBT in this module is NM `AOBT_3_flt`, never airport BLOCK_TIME.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from common import AIRPORTS, TRAIN_FILES, sec

# Shrinkage strengths from the training-DEP schema profile:
# stand×runway median n=37 → k=20 (half-weight near n=20).
# operator median n=10 → k=20 (strong shrink of rare carriers).
# stand×3h-bucket is denser than stand×hour (median hour-cell n=21) → k=25.
K_SR = 20.0
K_RW = 15.0
K_OP = 20.0
K_SH = 25.0

LOG_EPS = 1.0
HOUR_BUCKET_H = 3
CONG_WINDOWS_MIN = (10, 15)

# Feature lists consumed by the trainer. BLOCK_TIME / TAXITIME must never appear.
NUM_COLS = [
    "mvt_aobt",
    "aobt_sched",
    "aobt_eobt",
    "aobt_lobt",
    "aobt_iobt",
    "mvt_sched",
    "hour",
    "dow",
    "month",
    "hour_bucket",
    "is_holiday",
    "dep_rwy_aobt_10m",
    "dep_rwy_aobt_15m",
    "arr_rwy_10m",
    "arr_rwy_15m",
    "arr_other_rwy_10m",
    "arr_other_rwy_15m",
    "floor",
    "floor_n",
    "enc_stand_rwy",
    "enc_n_stand_rwy",
    "enc_operator",
    "enc_n_operator",
    "enc_stand_hour",
    "enc_n_stand_hour",
]

CAT_COLS = [
    "STAND_mvt",
    "RUNWAY_mvt",
    "AIRCRAFT_TYPE_mvt",
    "WK_TBL_CAT_flt",
    "MARKET_SEGMENT_flt",
    "AIRCRAFT_OPERATOR_flt",
    "FLIGHT_TYPE_flt",
    "ADES_mvt",
]

NUM_UNMATCHED = [
    "mvt_sched",
    "hour",
    "dow",
    "month",
    "hour_bucket",
    "is_holiday",
    "floor",
    "floor_n",
]
CAT_UNMATCHED = ["STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt"]

FORBIDDEN_FEATURE_SUBSTR = ("BLOCK_TIME", "TAXITIME", "BLOCK_TIME_UTC")

ENC_SPECS = (
    {
        "name": "stand_rwy",
        "keys": ["airport", "STAND_mvt", "RUNWAY_mvt"],
        "k": K_SR,
    },
    {
        "name": "operator",
        "keys": ["airport", "AIRCRAFT_OPERATOR_flt"],
        "k": K_OP,
    },
    {
        "name": "stand_hour",
        "keys": ["airport", "STAND_mvt", "hour_bucket"],
        "k": K_SH,
    },
)

# UTC dates of 2025 public holidays at the 10 airports. Close enough: Europe is
# UTC+1/+2, so a late-evening UTC date can be off by one local day — accepted.
_DE = {
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 5, 1),
    date(2025, 5, 29),
    date(2025, 6, 9),
    date(2025, 10, 3),
    date(2025, 12, 25),
    date(2025, 12, 26),
}
_BY = _DE | {date(2025, 1, 6), date(2025, 6, 19)}  # Bavaria (EDDM)
_UK = {
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 5, 5),
    date(2025, 5, 26),
    date(2025, 8, 25),
    date(2025, 12, 25),
    date(2025, 12, 26),
}
_NL = {
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 4, 26),
    date(2025, 5, 5),
    date(2025, 5, 29),
    date(2025, 6, 9),
    date(2025, 12, 25),
    date(2025, 12, 26),
}
_ES = {
    date(2025, 1, 1),
    date(2025, 1, 6),
    date(2025, 4, 18),
    date(2025, 5, 1),
    date(2025, 8, 15),
    date(2025, 10, 12),
    date(2025, 11, 1),
    date(2025, 12, 6),
    date(2025, 12, 8),
    date(2025, 12, 25),
}
_CT = _ES | {date(2025, 6, 24), date(2025, 9, 11)}  # Catalonia (LEBL)
_FR = {
    date(2025, 1, 1),
    date(2025, 4, 21),
    date(2025, 5, 1),
    date(2025, 5, 8),
    date(2025, 5, 29),
    date(2025, 6, 9),
    date(2025, 7, 14),
    date(2025, 8, 15),
    date(2025, 11, 1),
    date(2025, 11, 11),
    date(2025, 12, 25),
}
_IT = {
    date(2025, 1, 1),
    date(2025, 1, 6),
    date(2025, 4, 21),
    date(2025, 4, 25),
    date(2025, 5, 1),
    date(2025, 6, 2),
    date(2025, 8, 15),
    date(2025, 11, 1),
    date(2025, 12, 8),
    date(2025, 12, 25),
    date(2025, 12, 26),
}
_CH = {
    date(2025, 1, 1),
    date(2025, 4, 18),
    date(2025, 4, 21),
    date(2025, 5, 1),
    date(2025, 5, 29),
    date(2025, 6, 9),
    date(2025, 8, 1),
    date(2025, 12, 25),
    date(2025, 12, 26),
}
_TR = {
    date(2025, 1, 1),
    date(2025, 3, 30),
    date(2025, 3, 31),
    date(2025, 4, 1),
    date(2025, 4, 23),
    date(2025, 5, 1),
    date(2025, 5, 19),
    date(2025, 6, 6),
    date(2025, 6, 7),
    date(2025, 6, 8),
    date(2025, 6, 9),
    date(2025, 7, 15),
    date(2025, 8, 30),
    date(2025, 10, 29),
}

HOLIDAYS: dict[str, set[date]] = {
    "EDDF": _DE,
    "EDDM": _BY,
    "EGLL": _UK,
    "EHAM": _NL,
    "LEBL": _CT,
    "LEMD": _ES,
    "LFPG": _FR,
    "LIRF": _IT,
    "LSZH": _CH,
    "LTFM": _TR,
}


def assert_ranking_safe_features(cols: list[str] | tuple[str, ...]) -> None:
    bad = [c for c in cols if any(s in c for s in FORBIDDEN_FEATURE_SUBSTR)]
    if bad:
        raise RuntimeError(f"forbidden leakage columns in features: {bad}")


assert_ranking_safe_features(NUM_COLS + CAT_COLS + NUM_UNMATCHED + CAT_UNMATCHED)


def ns_array(s: pl.Series) -> np.ndarray:
    v = s.to_numpy()
    return v.astype("datetime64[ns]").astype(np.int64)


def _nat_ok(ns: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = ns.copy()
    out[~valid] = np.iinfo(np.int64).min
    return out


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------
def add_calendar(df: pl.DataFrame) -> pl.DataFrame:
    """Hour/dow from AOBT when present else MVT. Month stays MVT (split key)."""
    out = df.with_columns(
        pl.coalesce(pl.col("AOBT_3_flt").dt.hour(), pl.col("MVT_TIME_UTC_mvt").dt.hour()).alias("hour"),
        pl.coalesce(pl.col("AOBT_3_flt").dt.weekday(), pl.col("MVT_TIME_UTC_mvt").dt.weekday()).alias("dow"),
        (pl.coalesce(pl.col("AOBT_3_flt").dt.hour(), pl.col("MVT_TIME_UTC_mvt").dt.hour()) // HOUR_BUCKET_H).alias(
            "hour_bucket"
        ),
        pl.coalesce(pl.col("AOBT_3_flt").dt.date(), pl.col("MVT_TIME_UTC_mvt").dt.date()).alias("_cal_date"),
        pl.col("STAND_mvt").fill_null("__NA__"),
        pl.col("AIRCRAFT_OPERATOR_flt").fill_null("__NA__"),
        pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"),
    )
    ap = out["airport"].to_numpy()
    dates = out["_cal_date"].to_list()
    hol = np.zeros(out.height, dtype=np.int8)
    for i, (a, d) in enumerate(zip(ap, dates)):
        if d is None:
            continue
        if not isinstance(d, date):
            d = d.date() if hasattr(d, "date") else date.fromisoformat(str(d)[:10])
        hol[i] = 1 if d in HOLIDAYS.get(str(a), ()) else 0
    return out.with_columns(pl.Series("is_holiday", hol)).drop("_cal_date")


# ---------------------------------------------------------------------------
# Hierarchical p10 floor
# ---------------------------------------------------------------------------
def _p10_tables(hist: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """p10 + n at (airport,stand,runway), (airport,runway), (airport). Matched hist only."""
    h = hist.filter(~pl.col("unmatched")) if "unmatched" in hist.columns else hist
    h = h.filter(pl.col("y").is_finite())
    y = pl.col("y")
    aggs = [y.len().alias("n"), y.quantile(0.10).alias("p10")]
    return {
        "sr": h.group_by(["airport", "STAND_mvt", "RUNWAY_mvt"]).agg(aggs),
        "rw": h.group_by(["airport", "RUNWAY_mvt"]).agg(aggs),
        "ap": h.group_by(["airport"]).agg(aggs),
    }


def apply_floor(df: pl.DataFrame, tabs: dict[str, pl.DataFrame]) -> pl.DataFrame:
    sr = tabs["sr"].rename({"n": "sr_n", "p10": "sr_p10"})
    rw = tabs["rw"].rename({"n": "rw_n", "p10": "rw_p10"})
    ap = tabs["ap"].rename({"n": "ap_n", "p10": "ap_p10"})
    out = (
        df.join(sr, on=["airport", "STAND_mvt", "RUNWAY_mvt"], how="left")
        .join(rw, on=["airport", "RUNWAY_mvt"], how="left")
        .join(ap, on=["airport"], how="left")
    )
    n_sr = pl.col("sr_n").fill_null(0)
    n_rw = pl.col("rw_n").fill_null(0)
    p_ap = pl.col("ap_p10")
    p_rw = pl.col("rw_p10")
    p_sr = pl.col("sr_p10")
    rw_shrunk = (n_rw * p_rw + K_RW * p_ap) / (n_rw + K_RW)
    rw_shrunk = pl.when(n_rw > 0).then(rw_shrunk).otherwise(p_ap)
    sr_shrunk = (n_sr * p_sr + K_SR * rw_shrunk) / (n_sr + K_SR)
    floor = pl.when(n_sr > 0).then(sr_shrunk).otherwise(rw_shrunk)
    floor = floor.fill_null(p_ap).clip(lower_bound=1.0)
    return out.with_columns(floor.alias("floor"), n_sr.cast(pl.Float64).alias("floor_n")).drop(
        [c for c in ("sr_n", "sr_p10", "rw_n", "rw_p10", "ap_n", "ap_p10") if c in out.columns]
    )


def attach_floors_time_forward(train: pl.DataFrame, val: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Train floors from strictly earlier months; first train month is LOMO.

    Val floors use the entire training split (project convention: no val y).
    """
    months = sorted(int(m) for m in train["month"].unique().to_list())
    parts: list[pl.DataFrame] = []
    for i, m in enumerate(months):
        chunk = train.filter(pl.col("month") == m)
        earlier = train.filter(pl.col("month") < m)
        if earlier.height >= 500:
            hist = earlier
        else:
            hist = train.filter(pl.col("month") != m)
        parts.append(apply_floor(chunk, _p10_tables(hist)))
    tr = pl.concat(parts, how="vertical")
    va = apply_floor(val, _p10_tables(train))
    return tr, va


def attach_floors_full_train(train: pl.DataFrame, val: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ablation helper: same tables on train and val (train still has no self-row if OOF)."""
    tabs = _p10_tables(train)
    return apply_floor(train, tabs), apply_floor(val, tabs)


# ---------------------------------------------------------------------------
# James-Stein / empirical-Bayes target encodings (OOF)
# ---------------------------------------------------------------------------
def _js_table(hist: pl.DataFrame, keys: list[str], k: float) -> pl.DataFrame:
    h = hist.filter(~pl.col("unmatched")) if "unmatched" in hist.columns else hist
    h = h.filter(pl.col("y").is_finite())
    grp = h.group_by(keys).agg(pl.len().alias("_n"), pl.col("y").mean().alias("_mean"))
    prior = h.group_by(["airport"]).agg(pl.col("y").mean().alias("_prior"))
    t = grp.join(prior, on="airport", how="left")
    enc = (pl.col("_n") * pl.col("_mean") + k * pl.col("_prior")) / (pl.col("_n") + k)
    return t.select(*keys, enc.alias("_enc"), pl.col("_n").cast(pl.Float64).alias("_n"))


def _apply_js(df: pl.DataFrame, table: pl.DataFrame, keys: list[str], name: str, prior: pl.DataFrame) -> pl.DataFrame:
    j = df.join(table, on=keys, how="left").join(prior, on="airport", how="left")
    enc = pl.when(pl.col("_enc").is_not_null()).then(pl.col("_enc")).otherwise(pl.col("_prior"))
    n = pl.col("_n").fill_null(0.0)
    return j.with_columns(enc.alias(f"enc_{name}"), n.alias(f"enc_n_{name}")).drop(["_enc", "_n", "_prior"])


def _airport_prior(hist: pl.DataFrame) -> pl.DataFrame:
    h = hist.filter(~pl.col("unmatched")) if "unmatched" in hist.columns else hist
    return h.filter(pl.col("y").is_finite()).group_by("airport").agg(pl.col("y").mean().alias("_prior"))


def encode_with_tables(df: pl.DataFrame, tables: dict, prior: pl.DataFrame) -> pl.DataFrame:
    out = df
    for spec in ENC_SPECS:
        out = _apply_js(out, tables[spec["name"]], spec["keys"], spec["name"], prior)
    return out


def fit_encoding_tables(hist: pl.DataFrame) -> tuple[dict, pl.DataFrame]:
    tables = {spec["name"]: _js_table(hist, spec["keys"], spec["k"]) for spec in ENC_SPECS}
    return tables, _airport_prior(hist)


def attach_encodings_oof(train: pl.DataFrame, val: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Leave-one-month-out encodings on train; full-train tables on val."""
    months = sorted(int(m) for m in train["month"].unique().to_list())
    parts: list[pl.DataFrame] = []
    for m in months:
        hist = train.filter(pl.col("month") != m)
        if hist.height == 0:
            # Single-month train: 5-fold by row order, never the row's own fold.
            chunk = train.filter(pl.col("month") == m).with_row_index("_fold")
            chunk = chunk.with_columns((pl.col("_fold") % 5).alias("_fold"))
            sub = []
            for f in range(5):
                h = chunk.filter(pl.col("_fold") != f).drop("_fold")
                tables, prior = fit_encoding_tables(h if h.height else chunk.drop("_fold"))
                sub.append(encode_with_tables(chunk.filter(pl.col("_fold") == f).drop("_fold"), tables, prior))
            parts.append(pl.concat(sub, how="vertical"))
            continue
        tables, prior = fit_encoding_tables(hist)
        parts.append(encode_with_tables(train.filter(pl.col("month") == m), tables, prior))
    tr = pl.concat(parts, how="vertical")
    tables, prior = fit_encoding_tables(train)
    va = encode_with_tables(val, tables, prior)
    return tr, va


# ---------------------------------------------------------------------------
# Tight-window congestion (AOBT-indexed, same runway + arrival crossings)
# ---------------------------------------------------------------------------
def load_arr_rwy() -> pl.DataFrame:
    cols = ["ADES_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]
    return (
        pl.concat([pl.scan_parquet(p).select(cols) for p in TRAIN_FILES])
        .filter(pl.col("PHASE_mvt") == "ARR")
        .with_columns(pl.col("ADES_mvt").alias("airport"))
        .select("airport", "RUNWAY_mvt", "MVT_TIME_UTC_mvt")
        .collect()
        .sort(["airport", "MVT_TIME_UTC_mvt"])
    )


def add_congestion(dep: pl.DataFrame, arr: pl.DataFrame) -> pl.DataFrame:
    """Counts of other AOBTs / landings in the trailing 10 and 15 min.

    Query time = AOBT_3 if present else MVT (unmatched). Self AOBT excluded.
    Arrivals on a different runway at the same airport are the crossing proxy
    (no taxiway graph in this challenge).
    """
    n = dep.height
    ap = dep["airport"].to_numpy()
    rwy = dep["RUNWAY_mvt"].to_numpy().astype(str)
    aobt_valid = ~dep["AOBT_3_flt"].is_null().to_numpy()
    mvt_ns = ns_array(dep["MVT_TIME_UTC_mvt"])
    aobt_ns = _nat_ok(ns_array(dep["AOBT_3_flt"]), aobt_valid)
    q_ns = np.where(aobt_valid, aobt_ns, mvt_ns)

    arr_ap = arr["airport"].to_numpy()
    arr_rwy = arr["RUNWAY_mvt"].to_numpy().astype(str)
    arr_ns = ns_array(arr["MVT_TIME_UTC_mvt"])

    out = {f"dep_rwy_aobt_{w}m": np.zeros(n, dtype=np.float64) for w in CONG_WINDOWS_MIN}
    out.update({f"arr_rwy_{w}m": np.zeros(n, dtype=np.float64) for w in CONG_WINDOWS_MIN})
    out.update({f"arr_other_rwy_{w}m": np.zeros(n, dtype=np.float64) for w in CONG_WINDOWS_MIN})

    windows_ns = {w: int(w * 60 * 1e9) for w in CONG_WINDOWS_MIN}

    for a in AIRPORTS:
        dsel = np.where(ap == a)[0]
        if dsel.size == 0:
            continue
        asel = np.where(arr_ap == a)[0]
        d_q = q_ns[dsel]
        d_ao = aobt_ns[dsel]
        d_ok = aobt_valid[dsel]
        d_rw = rwy[dsel]

        ao_ok = d_ao[d_ok]
        rw_ok = d_rw[d_ok]
        a_t = arr_ns[asel] if asel.size else np.array([], dtype=np.int64)
        a_rw = arr_rwy[asel] if asel.size else np.array([], dtype=str)

        runways = np.unique(d_rw)
        for rw in runways:
            on_rw = d_rw == rw
            idx = dsel[on_rw]
            if idx.size == 0:
                continue
            q = d_q[on_rw]
            self_ao = d_ao[on_rw]
            self_ok = d_ok[on_rw]

            ao_same = np.sort(ao_ok[rw_ok == rw])
            ar_same = np.sort(a_t[a_rw == rw]) if a_t.size else np.array([], dtype=np.int64)
            ar_oth = np.sort(a_t[a_rw != rw]) if a_t.size else np.array([], dtype=np.int64)

            for w, wns in windows_ns.items():
                if ao_same.size:
                    hi = np.searchsorted(ao_same, q, side="right")
                    lo = np.searchsorted(ao_same, q - wns, side="left")
                    c = (hi - lo).astype(np.float64)
                    in_win = self_ok & (self_ao >= q - wns) & (self_ao <= q)
                    c -= in_win.astype(np.float64)
                    out[f"dep_rwy_aobt_{w}m"][idx] = np.maximum(c, 0.0)
                if ar_same.size:
                    hi = np.searchsorted(ar_same, q, side="right")
                    lo = np.searchsorted(ar_same, q - wns, side="left")
                    out[f"arr_rwy_{w}m"][idx] = (hi - lo).astype(np.float64)
                if ar_oth.size:
                    hi = np.searchsorted(ar_oth, q, side="right")
                    lo = np.searchsorted(ar_oth, q - wns, side="left")
                    out[f"arr_other_rwy_{w}m"][idx] = (hi - lo).astype(np.float64)

    extra = pl.DataFrame({"MVT_ID_mvt": dep["MVT_ID_mvt"], **out})
    return dep.join(extra, on="MVT_ID_mvt", how="left")


# ---------------------------------------------------------------------------
# Target transform / Duan smearing
# ---------------------------------------------------------------------------
def log_excess(y, floor) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    return np.log(np.clip(y - floor, LOG_EPS, None))


def smear_factor(z_true, z_pred) -> float:
    """Duan smearing: mean exp(residual) on a held-out *training* slice.

    Residuals clipped in log space so a handful of bombs cannot explode the
    back-transform. Smear itself is clipped to [1, 3].
    """
    z_true = np.asarray(z_true, dtype=np.float64)
    z_pred = np.asarray(z_pred, dtype=np.float64)
    m = np.isfinite(z_true) & np.isfinite(z_pred)
    if m.sum() < 10:
        return 1.0
    r = np.clip(z_true[m] - z_pred[m], -2.5, 2.5)
    s = float(np.mean(np.exp(r)))
    if not np.isfinite(s):
        return 1.0
    return float(np.clip(s, 1.0, 3.0))


def back_transform(z_pred, floor, smear: float) -> np.ndarray:
    z_pred = np.asarray(z_pred, dtype=np.float64)
    floor = np.asarray(floor, dtype=np.float64)
    yhat = floor + float(smear) * np.exp(z_pred)
    return np.clip(yhat, 1.0, 200_000.0)


def naive_exp_back(z_pred, floor) -> np.ndarray:
    return back_transform(z_pred, floor, 1.0)
