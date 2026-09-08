"""Apply a fitted temporal v2 model. LIRF unmatched stays on the E20 override."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import torch

from openair.models.temporal_v2.config import TemporalV2Config
from openair.models.temporal_v2.dataset import TaxiSequenceDataset
from openair.models.temporal_v2.model import TemporalV2Model
from openair.models.temporal_v2.train import predict_dataset


def load_model(ckpt_path: Path, vocabs: dict, device, mode: str) -> TemporalV2Model:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_d = blob.get("cfg", {})
    cfg = TemporalV2Config()
    for k, v in cfg_d.items():
        if hasattr(cfg, k) and k != "embed_dims":
            setattr(cfg, k, v)
    cfg.__post_init__()
    model = TemporalV2Model(cfg, vocabs, mode=blob.get("mode", mode))
    model.load_state_dict(blob["model"])
    model.to(device)
    model.eval()
    return model


def predict_correction(
    model: TemporalV2Model,
    ds: TaxiSequenceDataset,
    cfg: TemporalV2Config,
    device,
) -> np.ndarray:
    pred = predict_dataset(model, ds, cfg, device)
    e20 = ds.e20.astype(np.float64)
    corr = pred - e20
    corr[ds.lirf_override] = 0.0
    pred = e20 + corr
    pred[ds.lirf_override] = e20[ds.lirf_override]
    return pred, corr


def blend_grid(y: np.ndarray, e20: np.ndarray, temporal: np.ndarray, unmatched, airport) -> list[dict]:
    from openair.models.temporal_v2.metrics import score_block

    rows = []
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        p = w * e20 + (1.0 - w) * temporal
        s = score_block(y, p, unmatched, airport)
        s["w_e20"] = w
        rows.append(s)
    # residual add (w=1 on e20 plus correction already in temporal if residual)
    return rows
