"""Holdout metrics: overall / matched / tail SSE concentration."""
from __future__ import annotations

import numpy as np

from numpy.typing import ArrayLike


def _rmse(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[m] - p[m]) ** 2)))


def _mae(y, p) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    if not m.any():
        return float("nan")
    return float(np.mean(np.abs(y[m] - p[m])))


def score_block(y: ArrayLike, p: ArrayLike, unmatched: ArrayLike, airport: ArrayLike) -> dict:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    um = np.asarray(unmatched, dtype=bool)
    ap = np.asarray(airport)
    ok = np.isfinite(y) & np.isfinite(p)
    err2 = np.zeros_like(y)
    err2[ok] = (y[ok] - p[ok]) ** 2
    matched = ok & ~um
    abs_err = np.zeros_like(y)
    abs_err[ok] = np.abs(y[ok] - p[ok])
    total_sse = float(err2[ok].sum())
    n_ok = int(ok.sum())

    def sse_share(mask) -> float:
        if total_sse <= 0:
            return float("nan")
        return float(err2[ok & mask].sum() / total_sse)

    order = np.argsort(-abs_err)
    top1 = order[: max(1, int(0.01 * n_ok))]
    top5 = order[: max(1, int(0.05 * n_ok))]
    out = {
        "n": n_ok,
        "rmse": _rmse(y[ok], p[ok]),
        "mae": _mae(y[ok], p[ok]),
        "n_matched": int(matched.sum()),
        "rmse_matched": _rmse(y[matched], p[matched]) if matched.any() else float("nan"),
        "mae_matched": _mae(y[matched], p[matched]) if matched.any() else float("nan"),
        "sse": total_sse,
        "sse_top1pct": float(err2[top1].sum()),
        "sse_top5pct": float(err2[top5].sum()),
        "sse_share_top1pct": float(err2[top1].sum() / total_sse) if total_sse else float("nan"),
        "sse_share_top5pct": float(err2[top5].sum() / total_sse) if total_sse else float("nan"),
    }
    for sec, name in [
        (180, "gt180s"),
        (300, "gt300s"),
        (600, "gt600s"),
        (900, "gt900s"),
        (1800, "gt1800s"),
        (2700, "gt2700s"),
        (3600, "gt3600s"),
    ]:
        m = ok & (y > sec)
        out[f"n_{name}"] = int(m.sum())
        out[f"rmse_{name}"] = _rmse(y[m], p[m]) if m.any() else float("nan")
        out[f"sse_{name}"] = float(err2[m].sum())
        out[f"sse_share_{name}"] = sse_share(y > sec)
    for thr in (1800, 2700, 3600, 4500):
        m = matched & (y > thr)
        tag = f"matched_gt{thr // 60}m"
        out[f"n_{tag}"] = int(m.sum())
        out[f"rmse_{tag}"] = _rmse(y[m], p[m]) if m.any() else float("nan")
    for a in sorted(set(ap.tolist())):
        sel = ok & (ap == a)
        out[f"rmse_{a}"] = _rmse(y[sel], p[sel]) if sel.any() else float("nan")
    return out


def fmt_score(d: dict) -> str:
    return (
        f"rmse={d['rmse']:.2f}  matched={d['rmse_matched']:.2f}  mae={d['mae']:.2f}  "
        f">30m={d.get('rmse_matched_gt30m', float('nan')):.1f}  "
        f">60m={d.get('rmse_matched_gt60m', float('nan')):.1f}  "
        f"sse>1800={d.get('sse_share_gt1800s', float('nan')):.3f}"
    )
