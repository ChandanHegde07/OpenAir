"""Recompute E20 blends, decision and reports from saved OOF predictions.

Experts were already fit and saved to oof_predictions_{janjul,dec}.parquet.
This only re-runs the (cheap) blend/report stage.
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_e20_ensemble as e  # noqa: E402

RES = e.RES


def main():
    payload = {"generated_utc": e.datetime.now(e.timezone.utc).isoformat()}
    # reconstruct expert_scores and correlations from saved oof
    for split in ("janjul", "dec"):
        df = e.load_oof(split)
        y = np.asarray(df["y"].to_numpy())
        um = df["unmatched"].to_numpy().astype(bool)
        ap = df["airport"].to_numpy()
        scores = {}
        for k in e.EXPERTS:
            scores[k] = e.score_all(y, df[f"pred_{k}"].to_numpy(), um, ap)
        mm = (~um) & np.isfinite(y)
        rmat = np.column_stack([y[mm] - df[f"pred_{k}"].to_numpy()[mm] for k in e.EXPERTS])
        corr = e.pd.DataFrame(np.corrcoef(rmat.T), index=e.EXPERTS, columns=e.EXPERTS).round(4)
        amat = np.column_stack([np.abs(y[mm] - df[f"pred_{k}"].to_numpy()[mm]) for k in e.EXPERTS])
        acorr = e.pd.DataFrame(np.corrcoef(amat.T), index=e.EXPERTS, columns=e.EXPERTS).round(4)
        payload[split] = {"expert_scores": scores, "residual_corr": corr, "error_corr": acorr}

    blends = e.build_all_blends(payload)
    payload["blend_results"] = blends
    verdict, best, reason = e.decide_blend(blends)
    payload["verdict"], payload["best_blend"], payload["reason"] = verdict, best, reason
    e.log(f"VERDICT {verdict} best={best}")
    e.log(reason)

    # importance placeholder (needs models; not refit here) - read from E20.json if present
    pj = RES / "E20.json"
    if pj.exists():
        old = json.loads(pj.read_text())
        if "importance" in old.get("janjul", {}):
            payload["janjul"]["importance"] = old["janjul"]["importance"]

    e.make_reports(payload, blends)
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, e.pd.DataFrame):
            return clean(o.to_dict())
        if isinstance(o, e.pd.Series):
            return clean(o.to_dict())
        if isinstance(o, np.ndarray):
            return clean(o.tolist())
        if isinstance(o, (np.floating, float)):
            return float(o)
        if isinstance(o, (np.integer, int)) and not isinstance(o, bool):
            return int(o)
        if isinstance(o, (np.bool_, bool)):
            return bool(o)
        if o is None:
            return None
        return o

    (RES / "E20.json").write_text(json.dumps(clean(payload), indent=2), encoding="utf-8")
    import copy
    e.save_result("E20", clean(payload))
    e.write_summary(payload, blends)
    e.log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
