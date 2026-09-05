"""E11 follow-up: can ranking-safe MVT-SCHED identify LIRF unmatched extremes?"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    airport_mean_fallback,
    fill_with_fallback,
    fmt,
    load_dep,
    metrics_block,
    split_by_months,
)


def main():
    dep = load_dep()
    for split_name, months in {"janjul": [1, 7], "dec": [12]}.items():
        tr, va = split_by_months(dep, months)
        y = va["y"].to_numpy()
        um = va["unmatched"].to_numpy()
        ap = va["airport"].to_numpy()
        fb = airport_mean_fallback(tr, va)
        mvt_sched = va["mvt_sched"].to_numpy().astype(float)
        type_null = va["AIRCRAFT_TYPE_mvt"].is_null().to_numpy()
        # unmatched: use clip(mvt-sched)
        p_ms = np.clip(mvt_sched, 0, 14400)
        pred = fb.copy()
        pred[um] = np.where(np.isfinite(p_ms[um]), p_ms[um], fb[um])
        print(f"\n{split_name} unmatched -> clip(MVT-SCHED,0,4h) else ap_mean")
        print(" ", fmt(metrics_block(y, pred, um, ap)))
        # only LIRF unmatched
        pred2 = fb.copy()
        mask = um & (ap == "LIRF")
        pred2[mask] = np.where(np.isfinite(p_ms[mask]), p_ms[mask], fb[mask])
        print(f"{split_name} only LIRF unmatched -> clip(MVT-SCHED)")
        print(" ", fmt(metrics_block(y, pred2, um, ap)))
        # type-null at LIRF
        pred3 = fb.copy()
        mask = (ap == "LIRF") & type_null
        pred3[mask] = np.where(np.isfinite(p_ms[mask]), np.clip(mvt_sched[mask], 0, 14400), fb[mask])
        print(f"{split_name} LIRF type-null -> clip(MVT-SCHED)  n={mask.sum()}")
        print(" ", fmt(metrics_block(y, pred3, um, ap)))
        # correlation unmatched LIRF y vs mvt_sched
        m = um & (ap == "LIRF") & np.isfinite(mvt_sched)
        if m.sum():
            print(
                f"  LIRF unmatched corr(y, mvt_sched)={np.corrcoef(y[m], mvt_sched[m])[0,1]:.3f} "
                f"n={m.sum()} RMSE if predict mvt_sched={np.sqrt(np.mean((y[m]-mvt_sched[m])**2)):.1f} "
                f"RMSE clip4h={np.sqrt(np.mean((y[m]-np.clip(mvt_sched[m],0,14400))**2)):.1f} "
                f"RMSE ap_mean={np.sqrt(np.mean((y[m]-fb[m])**2)):.1f}"
            )
        # type null overall
        print(f"  type_null n={type_null.sum()} unmatched among them {um[type_null].mean():.3f}")
        print(f"  unmatched among type_null LIRF {um[(ap=='LIRF')&type_null].mean() if ((ap=='LIRF')&type_null).any() else 0:.3f}")


if __name__ == "__main__":
    main()
