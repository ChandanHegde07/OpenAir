"""E36 STEP 1 — where does v9 lose SSE on MATCHED rows?

v9 == E20 on matched rows (only LIRF-unmatched changed). Use cached E20 OOF.
Break matched SSE down by airport, D quantile, T quantile, hour, runway,
source disagreement, and surface. Find the top-SSE regime.
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E36"
RES.mkdir(parents=True, exist_ok=True)


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def brk(df, key, y, e, tag):
    rows = []
    for v in df[key].unique().to_list():
        if v is None or str(v) == "null":
            continue
        m = (df[key] == v).to_numpy()
        if m.sum() < 100:
            continue
        sse = np.sum(e[m] ** 2)
        rows.append((str(v), int(m.sum()), float(rmse(y[m], y[m] - e[m])), float(sse)))
    rows.sort(key=lambda t: -t[3])
    tot = sum(r[3] for r in rows)
    return [{"value": r[0], "n": r[1], "rmse": r[2], "sse": r[3], "sse_share": r[3] / tot} for r in rows[:12]]


def main():
    dep = load_dep().with_columns(
        (pl.col("MVT_TIME_UTC_mvt") - pl.col("SCHED_TIME_UTC_mvt")).dt.total_seconds().cast(pl.Float64).alias("D"),
        (pl.col("AOBT_3_flt") - pl.col("EOBT_1_flt")).dt.total_seconds().cast(pl.Float64).alias("sd_eobt"),
        pl.col("MVT_TIME_UTC_mvt").dt.weekday().cast(pl.Int64).alias("dow"),
    )
    payload = {}
    for split, months in {"janjul": [1, 7], "dec": [12]}.items():
        va = dep.filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(months))
        oof = pl.read_parquet(RES.parent / "E20" / f"oof_predictions_{split}.parquet")
        va = va.join(oof.select(["MVT_ID_mvt", "pred_C", "pred_D", "pred_E", "unmatched"]).rename({"unmatched": "_um"}), on="MVT_ID_mvt", how="left")
        e20 = 0.456 * va["pred_C"].to_numpy() + 0.053 * va["pred_D"].to_numpy() + 0.491 * va["pred_E"].to_numpy()
        y = va["y"].to_numpy().astype(float)
        um = va["_um"].to_numpy().astype(bool)
        m = ~um & np.isfinite(y) & np.isfinite(e20)
        vm = va.filter(pl.Series(m))
        ym, em = y[m], (y - e20)[m]
        sse = em ** 2
        out = {"n_matched": int(m.sum()), "matched_rmse": float(rmse(ym, ym - em)), "matched_sse": float(sse.sum())}
        # D quantile
        Dv = vm["D"].to_numpy().astype(float)
        qs = np.nanquantile(Dv, [0, .5, .75, .9, .95, .99, 1.0])
        rows = []
        for i in range(6):
            sel = (Dv >= qs[i]) & (Dv <= (qs[i + 1] if i < 5 else qs[6]))
            if sel.sum() < 100:
                continue
            rows.append({"bin": f"D_q{i}", "n": int(sel.sum()), "rmse": float(rmse(ym[sel], ym[sel] - em[sel])),
                         "sse_share": float(sse[sel].sum() / sse.sum())})
        out["D_quantiles"] = rows
        Ta = vm["y"].to_numpy().astype(float)
        tq = np.nanquantile(Ta, [0, .5, .75, .9, .95, .99, 1.0])
        rows = []
        for i in range(6):
            sel = (Ta >= tq[i]) & (Ta <= (tq[i + 1] if i < 5 else tq[6]))
            if sel.sum() < 100:
                continue
            rows.append({"bin": f"T_q{i}", "n": int(sel.sum()), "rmse": float(rmse(ym[sel], ym[sel] - em[sel])),
                         "sse_share": float(sse[sel].sum() / sse.sum())})
        out["T_quantiles"] = rows
        for key in ["airport", "RUNWAY_mvt", "hour", "dow"]:
            out[key] = brk(vm, key, ym, em, key)
        # source disagreement quantiles
        sdv = vm["sd_eobt"].to_numpy().astype(float)
        sq = np.nanquantile(sdv[np.isfinite(sdv)], [0, .5, .9, .99, 1.0])
        rows = []
        for i in range(4):
            sel = np.isfinite(sdv) & (sdv >= sq[i]) & (sdv <= sq[i + 1])
            if sel.sum() < 100:
                continue
            rows.append({"bin": f"sd_q{i}", "n": int(sel.sum()), "rmse": float(rmse(ym[sel], ym[sel] - em[sel])),
                         "sse_share": float(sse[sel].sum() / sse.sum())})
        out["source_disagreement"] = rows
        # top SSE concentration
        order = np.argsort(-sse)
        out["concentration"] = {f"top{k}": float(sse[order[:k]].sum() / sse.sum()) for k in (10, 50, 100, 500, int(0.01 * m.sum()), int(0.05 * m.sum()))}
        payload[split] = out
        log(f"== {split} == matched n={out['n_matched']:,} rmse={out['matched_rmse']:.1f} sse_share_top1%={out['concentration'][f'top{int(0.01*m.sum())}']:.3f}")
        log(f"   D quantiles {[(r['bin'], round(r['rmse']), round(r['sse_share'],2)) for r in out['D_quantiles']]}")
        log(f"   T quantiles {[(r['bin'], round(r['rmse']), round(r['sse_share'],2)) for r in out['T_quantiles']]}")
        log(f"   airports {[(r['value'], round(r['rmse']), round(r['sse_share'],2)) for r in out['airport'][:6]]}")
        log(f"   sd {[(r['bin'], round(r['rmse']), round(r['sse_share'],2)) for r in out['source_disagreement']]}")
    (RES / "E36_matched_sse.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES / "E36_matched_sse.json"))


if __name__ == "__main__":
    main()
