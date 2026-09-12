"""Standard matched-tail submission.

Splice a matched recipe onto v8 (LIRF unmatched G stays v8).

Usage:
  python experiments/submit.py              # v13 production recipe
  python experiments/submit.py --version v15 --hat-mode pos --operator
  python experiments/submit.py --version v16 --lam-hat 0.22 --lam-rec 0.5

Only change flags. Writes likable-eagle_{version}.parquet to submissions/,
repo root, and experiments/results/submit/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from matched_submit import (  # noqa: E402
    CAT,
    NUM,
    P90,
    HTHR,
    add_extra,
    fit_lgb,
    grec,
    load_matched_train,
    log,
    pack_and_write,
    pdf,
)
from common import ROOT  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Matched-tail splice onto v8")
    p.add_argument("--version", default="v15", help="output tag, e.g. v15")
    p.add_argument("--lam-rec", type=float, default=0.5, help="grec blend toward P+Δ")
    p.add_argument("--lam-hat", type=float, default=0.25, help="leftover residual vs e20")
    p.add_argument("--hat-mode", choices=["full", "pos", "clip", "gate"], default="pos")
    p.add_argument("--operator", action="store_true", default=True, help="add AIRCRAFT_OPERATOR_flt")
    p.add_argument("--no-operator", action="store_false", dest="operator")
    p.add_argument("--mix-q", type=float, default=0.0, help="mix quantile(0.65) Δ; 0 = L2 only (v14 used 0.3)")
    p.add_argument("--p90", type=float, default=P90)
    p.add_argument("--hthr", type=float, default=HTHR)
    p.add_argument("--cap900", action="store_true", help="use 900x63 Δ/hat capacity (E68/E70 best)")
    p.add_argument("--seeds", type=int, default=1, help="seed-average count for Δ/hat (E69/E70)")
    p.add_argument("--arr-taxiin", action="store_true", help="previous ARR taxi-in at same stand (ranking-safe)")
    p.add_argument("--arr-rich", action="store_true", help="ARR delay, AOBT-asof taxi-in, time since last DEP at stand")
    p.add_argument("--note", default="")
    return p.parse_args()


def main():
    args = parse_args()
    cats = list(CAT)
    if args.operator and "AIRCRAFT_OPERATOR_flt" not in cats:
        cats.append("AIRCRAFT_OPERATOR_flt")

    def X(df):
        p = df.select(NUM + extra_num + [c for c in cats if c in df.columns]).to_pandas()
        for c in cats:
            if c in p.columns:
                p[c] = p[c].fillna("NA").astype("category")
        return p

    extra_num = []
    arr = None
    if args.arr_taxiin or args.arr_rich:
        from matched_submit import attach_arr_stand, load_arr_stand
        extra_num = ["arr_taxiin", "since_arr"]
        if args.arr_rich:
            extra_num += [
                "arr_delay", "since_dep", "arr_taxiin_aobt", "since_arr_aobt",
                "arr_taxiin_2", "rwy_arr_taxiin",
            ]
            if "arr_type" not in cats:
                cats.append("arr_type")
        arr = load_arr_stand(rich=args.arr_rich)

    log("train matched...")
    tr = load_matched_train()
    if args.arr_taxiin or args.arr_rich:
        tr = attach_arr_stand(tr, arr, rich=args.arr_rich)
    if args.operator:
        tr = tr.with_columns(pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"))
    y = tr["y"].to_numpy().astype(float)
    P = np.clip(tr["mvt_aobt"].to_numpy().astype(float), 0, None)
    e_tr = tr["e20_pred"].to_numpy().astype(float)
    Xt = X(tr)
    import lightgbm as lgb
    from matched_submit import LGB_KW, SEED

    if args.cap900:
        LGB_KW = dict(LGB_KW, n_estimators=900, learning_rate=0.03, num_leaves=63, min_child_samples=40)

    def fit_models(target, seeds, alpha=None):
        models = []
        for s in seeds:
            kw = dict(LGB_KW, random_state=s)
            m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **kw) if alpha is not None else lgb.LGBMRegressor(**kw)
            m.fit(Xt, target, categorical_feature=[c for c in cats if c in Xt.columns])
            models.append(m)
        return models

    def predict(models, X):
        p = np.zeros(X.shape[0])
        for m in models:
            p += np.asarray(m.predict(X), float)
        return p / len(models)

    dseeds = ([SEED] if args.seeds <= 1 else list(range(1, args.seeds + 1)))
    hseeds = ([7] if args.seeds <= 1 else list(range(101, 101 + args.seeds)))

    log("  Δ + leftover hat")
    d_models = fit_models(y - P, dseeds)
    h_models = fit_models(y - e_tr, hseeds)
    dq_models = fit_models(y - P, dseeds, alpha=0.65) if args.mix_q > 0 else None

    log("ranking...")
    rank = pl.scan_parquet(ROOT / "data" / "ranking.parquet").filter(pl.col("PHASE_mvt") == "DEP").collect()
    rank = add_extra(rank)
    if args.arr_taxiin or args.arr_rich:
        rcols = [
            pl.col("ADES_mvt").alias("airport"),
            "STAND_mvt",
            pl.col("BLOCK_TIME_UTC_mvt").alias("aibt"),
            pl.col("TAXITIME_SEC_mvt").cast(pl.Float64).alias("arr_taxiin"),
        ]
        if args.arr_rich:
            rcols.extend(
                [
                    pl.col("SCHED_TIME_UTC_mvt").alias("sibt"),
                    "RUNWAY_mvt",
                    pl.col("AIRCRAFT_TYPE_mvt").alias("arr_type"),
                ]
            )
        rank_arr = (
            pl.scan_parquet(ROOT / "data" / "ranking.parquet")
            .filter(pl.col("PHASE_mvt") == "ARR")
            .select(rcols)
            .drop_nulls(["airport", "STAND_mvt", "aibt", "arr_taxiin"])
            .collect()
        )
        if args.arr_rich:
            rank_arr = rank_arr.with_columns((pl.col("aibt") - pl.col("sibt")).dt.total_seconds().alias("arr_delay"))
            rank_arr = rank_arr.sort(["airport", "STAND_mvt", "aibt"]).with_columns(
                pl.col("arr_taxiin").shift(1).over(["airport", "STAND_mvt"]).alias("arr_taxiin_2")
            )
        rank_arr = rank_arr.sort(["airport", "STAND_mvt", "aibt"])
        rank = attach_arr_stand(rank, rank_arr, rich=args.arr_rich)
    if args.operator:
        rank = rank.with_columns(pl.col("AIRCRAFT_OPERATOR_flt").fill_null("NA"))
    from matched_submit import _v8_path
    v8 = pl.read_parquet(_v8_path())
    matched = rank.filter(pl.col("AOBT_3_flt").is_not_null())
    matched = matched.join(v8.rename({"TAXITIME_SEC_mvt": "e20_pred"}), on="MVT_ID_mvt", how="left")
    Xm = X(matched)
    Pm = np.clip(matched["mvt_aobt"].to_numpy().astype(float), 0, None)
    dhat = predict(d_models, Xm)
    if dq_models is not None:
        dhat = (1 - args.mix_q) * dhat + args.mix_q * predict(dq_models, Xm)
    rec = np.clip(Pm + dhat, 0, None)
    hat = predict(h_models, Xm)
    e8 = matched["e20_pred"].to_numpy().astype(float)
    g = grec(e8, rec, args.lam_rec, args.p90, args.hthr)
    if args.hat_mode == "pos":
        h = np.maximum(hat, 0.0)
    elif args.hat_mode == "clip":
        h = np.clip(hat, -400, 900)
    elif args.hat_mode == "gate":
        h = hat * ((np.abs(hat) > 80) | (e8 > args.p90)).astype(float)
    else:
        h = hat
    new = np.maximum(g + args.lam_hat * h, 0.0)
    note = (
        f"grec λ={args.lam_rec} hat λ={args.lam_hat} mode={args.hat_mode} "
        f"cap900={args.cap900} seeds={args.seeds} "
        f"op={args.operator} mix_q={args.mix_q} {args.note}"
    )
    pack_and_write(args.version, matched["MVT_ID_mvt"].to_numpy(), new, note=note)


if __name__ == "__main__":
    main()
