"""Temporal v2 hyperparameters. Keep the TCN compact enough for ~2M DEP samples."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP_DIR = ROOT / "experiments"
RESULTS = EXP_DIR / "results" / "temporal_v2"
CACHE = RESULTS / "cache"
CKPT_DIR = RESULTS / "checkpoints"


SEQ_CATS = (
    "phase",
    "runway",
    "stand",
    "airline",
    "actype",
    "wtc",
)

CTX_CATS = (
    "airport",
    "runway",
    "stand",
    "airline",
    "actype",
    "wtc",
    "ades",
)

# Static per-movement numerics (ranking-safe; no TAXITIME / BLOCK).
SEQ_NUM_NAMES = (
    "is_dep",
    "is_arr",
    "unmatched",
    "type_null",
    "mvt_aobt",
    "aobt_eobt",
    "mvt_eobt",
    "mvt_sched",
    "hour_sin",
    "hour_cos",
    "prev_gap_log",
    "aobt_miss",
    "eobt_miss",
    "sched_miss",
)

REL_NUM_NAMES = (
    "dt_hours",
    "log_dt",
    "same_runway",
    "same_stand",
    "same_airline",
    "same_actype",
    "same_phase_dep",
)

CTX_NUM_NAMES = (
    "mvt_aobt",
    "aobt_eobt",
    "mvt_eobt",
    "mvt_sched",
    "geo_mean",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "unmatched",
    "type_null",
    "aobt_miss",
    "eobt_miss",
    "n_hist",
    "e20_pred",
)


@dataclass
class TemporalV2Config:
    seq_len: int = 64
    hidden: int = 96
    n_blocks: int = 5
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    kernel_size: int = 3
    dropout: float = 0.15
    embed_airport: int = 8
    embed_runway: int = 16
    embed_stand: int = 24
    embed_airline: int = 16
    embed_actype: int = 16
    embed_wtc: int = 4
    embed_phase: int = 4
    embed_ades: int = 16
    mlp_hidden: int = 256
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 8
    patience: int = 3
    grad_clip: float = 1.0
    aux_w_30: float = 0.12
    aux_w_60: float = 0.06
    seed: int = 1
    num_workers: int = 0
    amp: bool = True
    # Residual corrections are added to E20; clip at inference to protect the incumbent.
    corr_clip: float = 2400.0
    es_frac: float = 0.12

    embed_dims: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.embed_dims:
            self.embed_dims = {
                "phase": self.embed_phase,
                "airport": self.embed_airport,
                "runway": self.embed_runway,
                "stand": self.embed_stand,
                "airline": self.embed_airline,
                "actype": self.embed_actype,
                "wtc": self.embed_wtc,
                "ades": self.embed_ades,
            }
        if len(self.dilations) != self.n_blocks:
            self.dilations = tuple(2**i for i in range(self.n_blocks))


def ensure_dirs() -> None:
    for d in (RESULTS, CACHE, CKPT_DIR, RESULTS / "plots"):
        d.mkdir(parents=True, exist_ok=True)
