"""Train loop for temporal v2. Primary loss is taxi RMSE; aux heads are light."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from openair.models.temporal_v2.config import TemporalV2Config
from openair.models.temporal_v2.dataset import TaxiSequenceDataset, collate
from openair.models.temporal_v2.metrics import _rmse, score_block
from openair.models.temporal_v2.model import TemporalV2Model


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(ds: TaxiSequenceDataset, cfg: TemporalV2Config, shuffle: bool) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=False,
    )


def _loss(out: dict, batch: dict, cfg: TemporalV2Config) -> torch.Tensor:
    y = batch["y"]
    pred = out["pred"]
    keep = ~batch["lirf_override"]
    if keep.any():
        se = (pred[keep] - y[keep]) ** 2
        loss = se.mean()
    else:
        loss = (pred - y).pow(2).mean() * 0.0
    logits = out["aux_logits"]
    t30 = (y > 1800).float()
    t60 = (y > 3600).float()
    if keep.any() and cfg.aux_w_30 > 0:
        loss = loss + cfg.aux_w_30 * F.binary_cross_entropy_with_logits(logits[keep, 0], t30[keep])
    if keep.any() and cfg.aux_w_60 > 0:
        loss = loss + cfg.aux_w_60 * F.binary_cross_entropy_with_logits(logits[keep, 1], t60[keep])
    return loss


@torch.no_grad()
def predict_dataset(model: TemporalV2Model, ds: TaxiSequenceDataset, cfg: TemporalV2Config, device) -> np.ndarray:
    model.eval()
    loader = make_loader(ds, cfg, shuffle=False)
    pred = np.zeros(len(ds), dtype=np.float64)
    amp = cfg.amp and device.type == "cuda"
    for batch in loader:
        idx = batch.pop("index").numpy()
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", enabled=amp):
            out = model(batch)
        p = out["pred"].float()
        if model.mode == "residual":
            p = torch.where(batch["lirf_override"], batch["e20"], p)
            corr = p - batch["e20"]
            corr = corr.clamp(-cfg.corr_clip, cfg.corr_clip)
            p = batch["e20"] + corr
            p = torch.where(batch["lirf_override"], batch["e20"], p)
        pred[idx] = p.detach().cpu().numpy()
    return pred


def evaluate(model, ds, cfg, device, unmatched, airport) -> dict:
    p = predict_dataset(model, ds, cfg, device)
    return score_block(ds.y, p, unmatched, airport), p


def fit_model(
    model: TemporalV2Model,
    train_ds: TaxiSequenceDataset,
    es_ds: TaxiSequenceDataset,
    cfg: TemporalV2Config,
    device: torch.device,
    ckpt_path: Path,
    log=print,
) -> dict:
    set_seed(cfg.seed)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    train_loader = make_loader(train_ds, cfg, shuffle=True)
    best = float("inf")
    bad = 0
    history = []
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        amp = cfg.amp and device.type == "cuda"
        for batch in train_loader:
            batch.pop("index")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=amp):
                out = model(batch)
                loss = _loss(out, batch, cfg)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            running += float(loss.detach().cpu()) * batch["y"].size(0)
            n_seen += int(batch["y"].size(0))
        sched.step()
        es_pred = predict_dataset(model, es_ds, cfg, device)
        es_rmse = _rmse(es_ds.y, es_pred)
        row = {"epoch": epoch, "train_loss": running / max(n_seen, 1), "es_rmse": es_rmse}
        history.append(row)
        log(f"  epoch {epoch:02d}  loss={row['train_loss']:.1f}  es_rmse={es_rmse:.2f}")
        if es_rmse + 1e-6 < best:
            best = es_rmse
            bad = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "mode": model.mode,
                    "epoch": epoch,
                    "es_rmse": es_rmse,
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"  early stop at epoch {epoch}")
                break

    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(blob["model"])
    return {"best_es_rmse": best, "history": history, "ckpt": str(ckpt_path)}
