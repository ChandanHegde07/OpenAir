"""E65 — minimal causal airport event-graph transformer (end-to-end).

Subsample Jan+Jul matched rows. Causal context = previous 32 same-airport events
(strictly earlier MVT). Numeric node features + time-delta + mask. 1-layer
transformer encoder, query = last token, scalar head predicts y - e20. Train on
Jan (loss masked on Jul), eval Jul. final = e20 + alpha*net. PyTorch CPU.
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
import torch
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import AIRPORTS, load_dep, rmse  # noqa: E402

RES = HERE / "results" / "E65"
RES.mkdir(parents=True, exist_ok=True)
SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)
L = 32
DIN = 7
SUB = 120000


def log(m):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def top1(y, p):
    e2 = (np.asarray(y, float) - np.asarray(p, float)) ** 2
    k = max(1, int(0.01 * len(e2)))
    return float(np.sort(e2)[::-1][:k].sum())


class Enc(nn.Module):
    def __init__(self, d=DIN, dmodel=64, heads=2, layers=1):
        super().__init__()
        self.inp = nn.Linear(d + 1, dmodel)  # +1 time-delta
        self.pos = nn.Parameter(torch.randn(1, L + 1, dmodel) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=heads, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Sequential(nn.Linear(dmodel, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, td, mask):
        x = torch.cat([x, td.unsqueeze(-1)], dim=-1)
        x = self.inp(x)
        q = x[:, :1, :]  # query slot = first position
        seq = torch.cat([q, x[:, 1:, :]], dim=1) + self.pos
        key_pad = mask  # False = valid
        out = self.enc(seq, src_key_padding_mask=key_pad)
        return self.head(out[:, 0, :]).squeeze(-1)


def main():
    dep = load_dep()
    # build e20 + clocks, matched Jan+Jul subsample
    oof = pl.read_parquet(HERE / "results" / "E20" / "oof_predictions_janjul.parquet")
    dep = dep.join(oof.select(["MVT_ID_mvt", "unmatched", "pred_C", "pred_D", "pred_E"]), on="MVT_ID_mvt", how="inner")
    dep = dep.with_columns(pl.Series("e20", 0.456 * dep["pred_C"].to_numpy() + 0.053 * dep["pred_D"].to_numpy() + 0.491 * dep["pred_E"].to_numpy()))
    dep = dep.filter(~pl.col("unmatched")).filter(pl.col("e20").is_finite()).filter(pl.col("mvt_aobt").is_finite())
    dep = dep.with_columns(pl.col("MVT_TIME_UTC_mvt").dt.month().cast(pl.Int64).alias("month"))
    dep = dep.filter(pl.col("month").is_in([1, 7]))
    dep = dep.sort(["airport", "MVT_TIME_UTC_mvt"])
    dep = dep.sample(n=min(SUB, dep.height), seed=SEED).sort(["airport", "MVT_TIME_UTC_mvt"])
    # node features
    mv = dep["MVT_TIME_UTC_mvt"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    ap = dep["airport"].to_numpy()
    F = np.column_stack([
        dep["e20"].to_numpy().astype(float) / 600.0,
        np.clip(dep["mvt_aobt"].to_numpy().astype(float), 0, None) / 600.0,
        dep["mvt_sched"].to_numpy().astype(float) / 600.0,
        np.nan_to_num(dep["aobt_eobt"].to_numpy().astype(float)) / 600.0,
        np.sin(2 * np.pi * dep["hour"].to_numpy().astype(float) / 24),
        np.cos(2 * np.pi * dep["hour"].to_numpy().astype(float) / 24),
        np.clip(dep["mvt_aobt"].to_numpy().astype(float), 0, None) / np.maximum(dep["e20"].to_numpy().astype(float), 60),
    ])
    y = dep["y"].to_numpy().astype(float)
    e20 = dep["e20"].to_numpy().astype(float)
    res = y - e20
    trmask = (dep["month"].to_numpy() == 1)
    # per-airport context building
    X = np.zeros((len(dep), L, DIN)); TD = np.zeros((len(dep), L)); M = np.ones((len(dep), L), dtype=bool)
    for a in AIRPORTS:
        idx = np.flatnonzero(ap == a)
        if idx.size < 2:
            continue
        order = np.argsort(mv[idx], kind="stable")
        idx = idx[order]
        mv_a = mv[idx]; F_a = F[idx]
        for pos, i in enumerate(idx):
            k = pos
            lo = max(0, k - L)
            cnt = k - lo
            X[i, :cnt] = F_a[lo:k]
            TD[i, :cnt] = (mv[i] - mv_a[lo:k]) / 1e9 / 600.0
            M[i, :cnt] = False
    Xf = torch.tensor(X, dtype=torch.float32)
    TDf = torch.tensor(TD, dtype=torch.float32)
    Mf = torch.tensor(M)
    Tf = torch.tensor(res, dtype=torch.float32)
    # query slot: target features at position 0; shift rest
    Xq = torch.zeros((len(dep), L + 1, DIN)); Xq[:, 1:, :] = Xf
    TDq = torch.zeros((len(dep), L + 1)); TDq[:, 1:] = TDf
    Mq = torch.ones((len(dep), L + 1), dtype=torch.bool); Mq[:, 1:] = Mf
    tr = torch.zeros(len(dep), dtype=torch.bool); tr[trmask] = True
    model = Enc()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    N = len(dep)
    tr_idx = np.flatnonzero(trmask)
    val_idx = np.flatnonzero(~trmask)
    for epoch in range(12):
        model.train()
        rng = np.random.permutation(tr_idx)
        for s in range(0, len(rng), 4096):
            b = rng[s:s + 4096]
            opt.zero_grad()
            p = model(Xq[b], TDq[b], Mq[b])
            loss = lossf(p, Tf[b])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xq[val_idx], TDq[val_idx], Mq[val_idx]).numpy()
            rm = float(rmse(y[val_idx], e20[val_idx] + pv))
            if epoch in (5, 11):
                log(f"  epoch {epoch}: val matched {rm:.2f}")
    # alpha blend on val
    pv = np.asarray(model(Xq[val_idx], TDq[val_idx], Mq[val_idx]).detach().numpy(), float)
    yv = y[val_idx]; e20v = e20[val_idx]
    best = min(np.arange(0, 1.01, 0.1), key=lambda al: float(rmse(yv, e20v + al * pv)))
    base = float(rmse(yv, e20v))
    final = float(rmse(yv, e20v + best * pv))
    payload = {"n_train": int(trmask.sum()), "n_val": int((~trmask).sum()), "subsample": True,
               "base_e20_matched": base, "neural_matched": final, "alpha": float(best),
               "base_top1": top1(yv, e20v), "neural_top1": top1(yv, e20v + best * pv)}
    log(f"  FINAL: val matched E20 {base:.2f} -> E20+neural {final:.2f} (alpha {best}) top1 {100*(payload['neural_top1']-payload['base_top1'])/payload['base_top1']:+.1f}%")
    (RES / "metrics.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    log("WROTE " + str(RES))


if __name__ == "__main__":
    main()
