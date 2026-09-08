"""Airport-state TCN + target-flight context → taxi residual / direct prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from openair.models.temporal_v2.config import (
    CTX_CATS,
    CTX_NUM_NAMES,
    REL_NUM_NAMES,
    SEQ_CATS,
    SEQ_NUM_NAMES,
    TemporalV2Config,
)
from openair.models.temporal_v2.encoder import TCNEncoder


class CatEmbed(nn.Module):
    def __init__(self, vocabs: dict[str, dict], dims: dict[str, int], keys: tuple[str, ...]):
        super().__init__()
        self.keys = keys
        self.emb = nn.ModuleDict()
        self.out_dim = 0
        for k in keys:
            n = max(int(max(vocabs[k].values()) if vocabs[k] else 1) + 1, 2)
            d = int(dims[k])
            self.emb[k] = nn.Embedding(n, d, padding_idx=0)
            self.out_dim += d

    def forward(self, cat: torch.Tensor) -> torch.Tensor:
        # cat: (..., K)
        parts = [self.emb[k](cat[..., i].clamp(min=0)) for i, k in enumerate(self.keys)]
        return torch.cat(parts, dim=-1)


class TemporalV2Model(nn.Module):
    def __init__(self, cfg: TemporalV2Config, vocabs: dict[str, dict], mode: str = "residual"):
        super().__init__()
        if mode not in ("residual", "direct"):
            raise ValueError(mode)
        self.cfg = cfg
        self.mode = mode
        self.seq_emb = CatEmbed(vocabs, cfg.embed_dims, SEQ_CATS)
        self.ctx_emb = CatEmbed(vocabs, cfg.embed_dims, CTX_CATS)
        seq_in = self.seq_emb.out_dim + len(SEQ_NUM_NAMES) + len(REL_NUM_NAMES)
        self.encoder = TCNEncoder(seq_in, cfg.hidden, cfg.dilations, cfg.kernel_size, cfg.dropout)
        ctx_in = self.ctx_emb.out_dim + len(CTX_NUM_NAMES)
        self.ctx_mlp = nn.Sequential(
            nn.Linear(ctx_in, cfg.hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        fused = self.encoder.out_dim + cfg.hidden
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, cfg.mlp_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.mlp_hidden, 128),
            nn.GELU(),
            nn.Dropout(cfg.dropout * 0.5),
        )
        self.reg = nn.Linear(128, 1)
        self.aux = nn.Linear(128, 2)
        self._init()

    def _init(self) -> None:
        if self.mode == "residual":
            nn.init.zeros_(self.reg.weight)
            nn.init.zeros_(self.reg.bias)
        else:
            nn.init.xavier_uniform_(self.reg.weight)
            nn.init.constant_(self.reg.bias, 900.0)

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        seq_c = self.seq_emb(batch["seq_cat"])
        seq_x = torch.cat([seq_c, batch["seq_num"], batch["rel_num"]], dim=-1)
        state = self.encoder(seq_x, batch["mask"])
        ctx = torch.cat([self.ctx_emb(batch["ctx_cat"]), batch["ctx_num"]], dim=-1)
        ctx_h = self.ctx_mlp(ctx)
        h = self.head(torch.cat([state, ctx_h], dim=-1))
        raw = self.reg(h).squeeze(-1)
        if self.mode == "residual":
            pred = batch["e20"] + raw
        else:
            pred = raw
        return {
            "pred": pred,
            "correction": raw if self.mode == "residual" else pred - batch["e20"],
            "aux_logits": self.aux(h),
        }
