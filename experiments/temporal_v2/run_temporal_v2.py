"""OpenAir v2 — TCN airport state on top of E20.

Baselines: E20 alone, temporal-alone, E20 + temporal residual, small blend grid.
Fits and vocabs use training_*.parquet only. No ranking/submitting.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from common import load_dep, split_by_months  # noqa: E402
from openair.models.e20_baseline.fit import load_cached_e20_oof  # noqa: E402
from openair.models.temporal_v2.config import (  # noqa: E402
    CTX_NUM_NAMES,
    SEQ_NUM_NAMES,
    CKPT_DIR,
    RESULTS,
    TemporalV2Config,
    ensure_dirs,
)
from openair.models.temporal_v2.dataset import (  # noqa: E402
    PackedMovementStore,
    TaxiSequenceDataset,
    attach_target_aliases,
    fit_scaler,
    fit_vocabs,
)
from openair.models.temporal_v2.metrics import fmt_score, score_block  # noqa: E402
from openair.models.temporal_v2.model import TemporalV2Model  # noqa: E402
from openair.models.temporal_v2.movements import load_training_movements  # noqa: E402
from openair.models.temporal_v2.train import evaluate, fit_model, predict_dataset, set_seed  # noqa: E402
from run_e2b_e3 import attach_geometry, geometry_tables  # noqa: E402

SPLITS = {"janjul": [1, 7], "dec": [12]}


def log(m: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def json_conv(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def time_tail_split(df: pl.DataFrame, frac: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    d = df.sort("t_sec")
    n = d.height
    k = int(n * (1.0 - frac))
    k = min(max(k, 1), n - 1)
    return d.head(k), d.tail(n - k)


def attach_geo(tr0: pl.DataFrame, va0: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    tabs = geometry_tables(tr0.filter(~pl.col("unmatched")))
    return attach_geometry(tr0, tabs, 30), attach_geometry(va0, tabs, 30)


def join_e20(df: pl.DataFrame, oof: pl.DataFrame | None) -> pl.DataFrame:
    if "e20_pred" in df.columns:
        df = df.drop("e20_pred")
    if oof is None:
        return df.with_columns(pl.lit(None).cast(pl.Float64).alias("e20_pred"))
    return df.join(oof.select("MVT_ID_mvt", "e20_pred"), on="MVT_ID_mvt", how="left")


def save_scores(path: Path, rows: list[dict]) -> None:
    pl.DataFrame(rows).write_csv(path)


def plot_rmse(rows: list[dict], path: Path) -> None:
    names = [r["model"] for r in rows]
    overall = [r["rmse"] for r in rows]
    matched = [r["rmse_matched"] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.18, overall, 0.36, label="overall")
    ax.bar(x + 0.18, matched, 0.36, label="matched")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("RMSE (s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run_split(args, cfg: TemporalV2Config, device, mov: pl.DataFrame, dep: pl.DataFrame) -> dict:
    val_months = SPLITS[args.split]
    log(f"========== {args.split} val months={val_months} ==========")
    tr0, va0 = split_by_months(dep, val_months)
    tr0, va0 = attach_geo(tr0, va0)
    oof_va = load_cached_e20_oof(args.split)
    va = join_e20(va0, oof_va)
    oof_tr_path = RESULTS / f"e20_oof_train_{args.split}.parquet"
    oof_tr = pl.read_parquet(oof_tr_path) if oof_tr_path.exists() else None
    if oof_tr is None:
        log(f"  no train OOF at {oof_tr_path} — residual model skipped")
    tr = join_e20(tr0, oof_tr)

    if args.airports:
        keep = [a.strip() for a in args.airports.split(",") if a.strip()]
        tr = tr.filter(pl.col("airport").is_in(keep))
        va = va.filter(pl.col("airport").is_in(keep))
        mov_s = mov.filter(pl.col("airport").is_in(keep))
        log(f"  airport filter {keep}: train={tr.height:,} val={va.height:,} mov={mov_s.height:,}")
    else:
        mov_s = mov

    n_e20_va = int(va["e20_pred"].is_finite().sum())
    if n_e20_va != va.height:
        raise RuntimeError(f"val E20 OOF missing {va.height - n_e20_va} rows")
    e20_va = va["e20_pred"].to_numpy()
    y_va = va["y"].to_numpy().astype(np.float64)
    um_va = va["unmatched"].to_numpy().astype(bool)
    ap_va = va["airport"].to_numpy()
    e20_score = score_block(y_va, e20_va, um_va, ap_va)
    log(f"  E20 holdout {fmt_score(e20_score)}")

    tr_fit, tr_es = time_tail_split(tr, cfg.es_frac)
    if args.max_train and tr_fit.height > args.max_train:
        tr_fit = tr_fit.sort("t_sec").tail(args.max_train)
        log(f"  max_train clipped fit to {tr_fit.height:,}")

    vocabs = fit_vocabs(mov_s.filter(~pl.col("month").is_in(val_months)), tr_fit)
    seq_scaler = fit_scaler(mov_s.filter(~pl.col("month").is_in(val_months)), SEQ_NUM_NAMES)
    ctx_direct = tr_fit.with_columns(pl.lit(0.0).alias("e20_pred"), pl.lit(0.0).alias("n_hist"))
    ctx_scaler_d = fit_scaler(ctx_direct, CTX_NUM_NAMES)
    log("  building packed movement store...")
    store = PackedMovementStore(mov_s, vocabs, seq_scaler)

    def make_ds(frame: pl.DataFrame, require_e20: bool, zero_e20: bool, scaler) -> TaxiSequenceDataset:
        return TaxiSequenceDataset(
            store, frame, vocabs, scaler, cfg, require_e20=require_e20, zero_e20_context=zero_e20
        )

    log("  datasets (direct)...")
    ds_tr_d = make_ds(tr_fit, False, True, ctx_scaler_d)
    ds_es_d = make_ds(tr_es, False, True, ctx_scaler_d)
    ds_va_d = make_ds(va, False, True, ctx_scaler_d)
    leak = ds_va_d.leakage_sample(2000)
    log(f"  leakage audit val: {leak}")
    if not leak["ok"]:
        raise RuntimeError(f"sequence leakage detected: {leak}")
    leak_tr = ds_tr_d.leakage_sample(2000)
    log(f"  leakage audit train: {leak_tr}")
    if not leak_tr["ok"]:
        raise RuntimeError(f"train sequence leakage: {leak_tr}")

    bundle = {
        "vocabs": vocabs,
        "seq_scaler": seq_scaler,
        "ctx_scaler_direct": ctx_scaler_d,
        "cfg": cfg,
        "split": args.split,
        "val_months": val_months,
    }
    (RESULTS / f"bundle_{args.split}.pkl").write_bytes(pickle.dumps(bundle))

    payload = {
        "split": args.split,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "e20": e20_score,
        "leakage_val": leak,
        "n_train_fit": tr_fit.height,
        "n_train_es": tr_es.height,
        "n_val": va.height,
        "device": str(device),
        "seq_len": cfg.seq_len,
        "hidden": cfg.hidden,
    }
    scores = [{"model": "E20", **e20_score}]
    preds_va = {"e20": e20_va}

    if not args.skip_direct:
        log("  train temporal DIRECT (predict TAXITIME)...")
        model_d = TemporalV2Model(cfg, vocabs, mode="direct")
        ckpt_d = CKPT_DIR / f"{args.split}_direct.pt"
        hist_d = fit_model(model_d, ds_tr_d, ds_es_d, cfg, device, ckpt_d, log=log)
        sc_d, p_d = evaluate(model_d, ds_va_d, cfg, device, um_va, ap_va)
        log(f"  temporal-alone {fmt_score(sc_d)}")
        payload["direct_history"] = hist_d
        payload["temporal_alone"] = sc_d
        scores.append({"model": "temporal_alone", **sc_d})
        preds_va["temporal_alone"] = p_d
        for w in (0.5, 0.75, 0.9):
            p = w * e20_va + (1.0 - w) * p_d
            sc = score_block(y_va, p, um_va, ap_va)
            scores.append({"model": f"blend_w{w}", **sc})
            log(f"  blend w_e20={w} {fmt_score(sc)}")
            preds_va[f"blend_w{w}"] = p

    resid_ready = oof_tr is not None and int(tr_fit["e20_pred"].is_finite().sum()) > 5000
    if args.skip_residual:
        resid_ready = False
    if resid_ready:
        tr_res = tr_fit.filter(pl.col("e20_pred").is_finite())
        es_res = tr_es.filter(pl.col("e20_pred").is_finite())
        log(f"  residual train rows with OOF E20: fit={tr_res.height:,} es={es_res.height:,}")
        ctx_res = tr_res.with_columns(pl.lit(0.0).alias("n_hist"))
        ctx_scaler_r = fit_scaler(ctx_res, CTX_NUM_NAMES)
        bundle["ctx_scaler_resid"] = ctx_scaler_r
        (RESULTS / f"bundle_{args.split}.pkl").write_bytes(pickle.dumps(bundle))
        ds_tr_r = make_ds(tr_res, True, False, ctx_scaler_r)
        ds_es_r = make_ds(es_res, True, False, ctx_scaler_r)
        ds_va_r = make_ds(va, True, False, ctx_scaler_r)
        log("  train temporal RESIDUAL (E20 correction)...")
        model_r = TemporalV2Model(cfg, vocabs, mode="residual")
        ckpt_r = CKPT_DIR / f"{args.split}_residual.pt"
        hist_r = fit_model(model_r, ds_tr_r, ds_es_r, cfg, device, ckpt_r, log=log)
        sc_r, p_r = evaluate(model_r, ds_va_r, cfg, device, um_va, ap_va)
        log(f"  E20+temporal {fmt_score(sc_r)}")
        payload["residual_history"] = hist_r
        payload["e20_plus_temporal"] = sc_r
        scores.append({"model": "E20_plus_temporal", **sc_r})
        preds_va["E20_plus_temporal"] = p_r
        # tiny shrink of the correction
        corr = p_r - e20_va
        for a in (0.5, 0.75, 1.0):
            p = e20_va + a * corr
            sc = score_block(y_va, p, um_va, ap_va)
            scores.append({"model": f"E20_plus_{a}corr", **sc})
            log(f"  E20+{a}*corr {fmt_score(sc)}")
            preds_va[f"E20_plus_{a}corr"] = p
    else:
        log("  residual path not run")

    best = min(scores, key=lambda r: r["rmse"])
    payload["scores"] = scores
    payload["best_model"] = best["model"]
    payload["best_rmse"] = best["rmse"]
    payload["delta_vs_e20"] = best["rmse"] - e20_score["rmse"]
    log(f"  BEST {best['model']} rmse={best['rmse']:.2f} Δ vs E20 {payload['delta_vs_e20']:+.2f}")

    out_dir = RESULTS / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    save_scores(out_dir / "scores.csv", scores)
    plot_rmse(scores, out_dir / "rmse_comparison.png")
    pred_df = pl.DataFrame({"MVT_ID_mvt": va["MVT_ID_mvt"], "y": y_va, "unmatched": um_va, "airport": ap_va})
    for k, v in preds_va.items():
        pred_df = pred_df.with_columns(pl.Series(k, np.asarray(v, dtype=np.float64)))
    pred_df.write_parquet(out_dir / "val_predictions.parquet")
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=json_conv), encoding="utf-8")
    lines = [
        f"# Temporal v2 — {args.split}",
        "",
        f"E20: {fmt_score(e20_score)}",
        "",
        "| Model | Overall RMSE | Matched | MAE | >30m matched | >60m matched |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in scores:
        lines.append(
            f"| {r['model']} | {r['rmse']:.2f} | {r['rmse_matched']:.2f} | {r['mae']:.2f} | "
            f"{r.get('rmse_matched_gt30m', float('nan')):.1f} | {r.get('rmse_matched_gt60m', float('nan')):.1f} |"
        )
    lines += ["", f"Best: **{best['model']}**  Δ vs E20 {payload['delta_vs_e20']:+.2f} s", ""]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"  wrote {out_dir}")
    return payload


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["janjul", "dec", "both"], default="janjul")
    ap.add_argument("--sanity", action="store_true", help="tiny LSZH run to verify the pipeline")
    ap.add_argument("--airports", type=str, default="")
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--skip-direct", action="store_true")
    ap.add_argument("--skip-residual", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cfg = TemporalV2Config(seq_len=args.seq_len, hidden=args.hidden, batch_size=args.batch_size)
    if args.epochs:
        cfg.epochs = args.epochs
    if args.sanity:
        args.airports = args.airports or "LSZH"
        args.max_train = args.max_train or 12000
        cfg.epochs = args.epochs or 2
        cfg.seq_len = min(cfg.seq_len, 32)
        cfg.hidden = 64
        cfg.n_blocks = 4
        cfg.dilations = (1, 2, 4, 8)
        cfg.batch_size = 128
        cfg.patience = 2
        log("SANITY mode: LSZH, 12k train, 2 epochs, seq=32")
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}  seq={cfg.seq_len} hidden={cfg.hidden} batch={cfg.batch_size}")

    log("load training movements (DEP+ARR, ranking-safe clocks only)...")
    mov = load_training_movements()
    log(f"  movements {mov.height:,}")
    log("load DEP targets...")
    dep = attach_target_aliases(
        load_dep().with_columns(pl.col("AIRCRAFT_TYPE_mvt").is_null().cast(pl.Int8).alias("type_null"))
    )
    log(f"  DEP {dep.height:,}")

    splits = ["janjul", "dec"] if args.split == "both" else [args.split]
    all_payload = {}
    for sp in splits:
        args.split = sp
        all_payload[sp] = run_split(args, cfg, device, mov, dep)
    (RESULTS / "temporal_v2.json").write_text(json.dumps(all_payload, indent=2, default=json_conv), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
