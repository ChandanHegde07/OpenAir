"""Vectorized historical-sequence construction. Chronological, strictly t_hist < t."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from openair.models.temporal_v2.config import (
    CTX_CATS,
    CTX_NUM_NAMES,
    REL_NUM_NAMES,
    SEQ_CATS,
    SEQ_NUM_NAMES,
    TemporalV2Config,
)

PAD = 0
UNK = 1


def attach_target_aliases(df: pl.DataFrame) -> pl.DataFrame:
    """Context column names expected by the TCN, derived from load_dep()."""
    import math

    two_pi = 2.0 * math.pi
    return df.with_columns(
        pl.col("RUNWAY_mvt").alias("runway"),
        pl.col("STAND_mvt").alias("stand"),
        pl.col("AIRCRAFT_OPERATOR_flt").alias("airline"),
        pl.col("AIRCRAFT_TYPE_mvt").alias("actype"),
        pl.col("WK_TBL_CAT_flt").alias("wtc"),
        pl.col("ADES_mvt").alias("ades"),
        pl.col("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t_sec"),
        ((pl.col("hour").cast(pl.Float32) * (two_pi / 24.0)).sin()).alias("hour_sin"),
        ((pl.col("hour").cast(pl.Float32) * (two_pi / 24.0)).cos()).alias("hour_cos"),
        ((pl.col("dow").cast(pl.Float32) * (two_pi / 7.0)).sin()).alias("dow_sin"),
        ((pl.col("dow").cast(pl.Float32) * (two_pi / 7.0)).cos()).alias("dow_cos"),
        pl.col("AOBT_3_flt").is_null().cast(pl.Float32).alias("aobt_miss"),
        pl.col("EOBT_1_flt").is_null().cast(pl.Float32).alias("eobt_miss"),
    )


def fit_vocab(series: pl.Series, min_count: int = 5) -> dict[str, int]:
    vc = series.fill_null("__NULL__").value_counts()
    names = vc[series.name].to_list() if series.name in vc.columns else vc[vc.columns[0]].to_list()
    counts = vc["count"].to_list()
    vocab = {"__PAD__": PAD, "__UNK__": UNK}
    i = 2
    for name, c in sorted(zip(names, counts), key=lambda t: (-t[1], str(t[0]))):
        key = "__NULL__" if name is None else str(name)
        if c < min_count:
            continue
        if key not in vocab:
            vocab[key] = i
            i += 1
    return vocab


def encode_col(series: pl.Series, vocab: dict[str, int]) -> np.ndarray:
    mapping = pl.DataFrame(
        {"k": [str(k) for k in vocab.keys()], "id": np.fromiter(vocab.values(), dtype=np.int32)}
    )
    keys = series.fill_null("__NULL__").cast(pl.Utf8).fill_null("__NULL__").alias("k")
    mapped = pl.DataFrame({"k": keys}).join(mapping, on="k", how="left")
    return mapped["id"].fill_null(UNK).to_numpy().astype(np.int32)


def fit_vocabs(train_mov: pl.DataFrame, train_tgt: pl.DataFrame) -> dict[str, dict[str, int]]:
    vocabs = {
        "phase": {"__PAD__": PAD, "__UNK__": UNK, "ARR": 2, "DEP": 3},
        "runway": fit_vocab(train_mov["runway"], 3),
        "stand": fit_vocab(train_mov["stand"], 8),
        "airline": fit_vocab(train_mov["airline"], 8),
        "actype": fit_vocab(train_mov["actype"], 5),
        "wtc": fit_vocab(train_mov["wtc"], 1),
        "airport": fit_vocab(train_tgt["airport"], 1),
        "ades": fit_vocab(train_tgt["ADES_mvt"], 20),
    }
    return vocabs


@dataclass
class NumScaler:
    mean: np.ndarray
    std: np.ndarray
    names: tuple[str, ...]

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = np.asarray(x, dtype=np.float32).copy()
        mean = self.mean.astype(np.float32, copy=False)
        std = self.std.astype(np.float32, copy=False)
        z -= mean
        z /= std
        z[~np.isfinite(z)] = 0.0
        np.clip(z, -8.0, 8.0, out=z)
        return z


def fit_scaler(df: pl.DataFrame, names: tuple[str, ...]) -> NumScaler:
    mean = np.zeros(len(names), dtype=np.float64)
    std = np.ones(len(names), dtype=np.float64)
    for i, name in enumerate(names):
        if name not in df.columns:
            continue
        s = df[name].cast(pl.Float64, strict=False)
        m = s.mean()
        sd = s.std()
        mean[i] = float(m) if m is not None and np.isfinite(m) else 0.0
        std[i] = float(sd) if sd is not None and np.isfinite(sd) and sd > 1e-6 else 1.0
    return NumScaler(mean=mean, std=std, names=names)


def _num_matrix(df: pl.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    n = df.height
    out = np.zeros((n, len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        if name not in df.columns:
            continue
        v = df[name].to_numpy()
        out[:, i] = np.asarray(v, dtype=np.float32)
    return out


class PackedMovementStore:
    """All movements packed with index 0 = PAD. Per-airport contiguous slices."""

    def __init__(self, mov: pl.DataFrame, vocabs: dict[str, dict], seq_scaler: NumScaler):
        n = mov.height
        self.n = n
        # row 0 is PAD
        self.times = np.zeros(n + 1, dtype=np.int64)
        self.times[1:] = mov["t_sec"].to_numpy().astype(np.int64)
        self.mvt_id = np.zeros(n + 1, dtype=np.float64)
        self.mvt_id[1:] = mov["MVT_ID_mvt"].to_numpy().astype(np.float64)
        self.airport = np.array([""] + mov["airport"].to_list(), dtype=object)

        cat = np.zeros((n + 1, len(SEQ_CATS)), dtype=np.int32)
        for j, name in enumerate(SEQ_CATS):
            cat[1:, j] = encode_col(mov[name], vocabs[name])
        self.cat = cat

        raw = _num_matrix(mov, SEQ_NUM_NAMES)
        scaled = seq_scaler.transform(raw)
        num = np.zeros((n + 1, len(SEQ_NUM_NAMES)), dtype=np.float32)
        num[1:] = scaled
        self.num = num

        rwy = np.zeros(n + 1, dtype=np.int32)
        stand = np.zeros(n + 1, dtype=np.int32)
        airline = np.zeros(n + 1, dtype=np.int32)
        actype = np.zeros(n + 1, dtype=np.int32)
        rwy[1:] = cat[1:, SEQ_CATS.index("runway")]
        stand[1:] = cat[1:, SEQ_CATS.index("stand")]
        airline[1:] = cat[1:, SEQ_CATS.index("airline")]
        actype[1:] = cat[1:, SEQ_CATS.index("actype")]
        self.rwy = rwy
        self.stand = stand
        self.airline = airline
        self.actype = actype
        self.is_dep = np.zeros(n + 1, dtype=np.int8)
        self.is_dep[1:] = mov["is_dep"].to_numpy().astype(np.int8)

        starts = {}
        pos = 1
        aps = np.asarray(mov["airport"].to_list())
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = aps[1:] != aps[:-1]
        cuts = np.flatnonzero(change)
        cuts = np.append(cuts, n)
        for a, b in zip(cuts[:-1], cuts[1:]):
            starts[str(aps[a])] = (pos + int(a), pos + int(b))
        self.ap_range = starts

        self._id_map = pl.DataFrame(
            {"MVT_ID_mvt": mov["MVT_ID_mvt"].to_numpy().astype(np.float64), "pos": np.arange(1, n + 1, dtype=np.int32)}
        )

    def locate(self, mvt_ids: np.ndarray) -> np.ndarray:
        j = pl.DataFrame({"MVT_ID_mvt": np.asarray(mvt_ids, dtype=np.float64), "_i": np.arange(len(mvt_ids))})
        j = j.join(self._id_map, on="MVT_ID_mvt", how="left").sort("_i")
        pos = j["pos"].to_numpy()
        if int(j["pos"].null_count()) != 0:
            raise RuntimeError(f"{int(j['pos'].null_count())} target MVT_IDs missing from movement store")
        return pos.astype(np.int32)


def build_seq_index(
    pos: np.ndarray,
    ap_start: np.ndarray,
    times: np.ndarray,
    target_times: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Previous seq_len movements with event_time < target_time. Vectorized."""
    offsets = np.arange(seq_len, 0, -1, dtype=np.int32)
    seq_idx = pos.astype(np.int32)[:, None] - offsets[None, :]
    valid = seq_idx >= ap_start.astype(np.int32)[:, None]
    clipped = np.where(valid, seq_idx, 0)
    valid = valid & (times[clipped] < target_times.astype(np.int64)[:, None])
    # never include the target row itself
    valid = valid & (clipped != pos.astype(np.int32)[:, None])
    clipped = np.where(valid, clipped, 0).astype(np.int32)
    return clipped, valid.astype(np.bool_)


class TaxiSequenceDataset(Dataset):
    def __init__(
        self,
        store: PackedMovementStore,
        targets: pl.DataFrame,
        vocabs: dict[str, dict],
        ctx_scaler: NumScaler,
        cfg: TemporalV2Config,
        require_e20: bool = False,
        zero_e20_context: bool = False,
    ):
        self.store = store
        self.cfg = cfg
        tgt = targets
        if require_e20:
            tgt = tgt.filter(pl.col("e20_pred").is_not_null() & pl.col("e20_pred").is_finite())
        n = tgt.height
        if n == 0:
            raise RuntimeError("no targets for dataset")
        self.n = n
        self.y = tgt["y"].to_numpy().astype(np.float32)
        self.e20 = tgt["e20_pred"].fill_null(0.0).to_numpy().astype(np.float32)
        self.unmatched = tgt["unmatched"].to_numpy().astype(np.bool_)
        ap = tgt["airport"].to_numpy()
        self.lirf_override = self.unmatched & (ap == "LIRF")
        self.mvt_id = tgt["MVT_ID_mvt"].to_numpy().astype(np.float64)
        self.airport_str = np.asarray(ap, dtype=object)

        pos = store.locate(self.mvt_id)
        self.pos = pos
        self.tgt_time = tgt["t_sec"].to_numpy().astype(np.int64) if "t_sec" in tgt.columns else store.times[pos]
        ap_start = np.array([store.ap_range[str(a)][0] for a in ap], dtype=np.int32)
        self.seq_idx, self.mask = build_seq_index(
            pos, ap_start, store.times, self.tgt_time, cfg.seq_len
        )
        n_hist = self.mask.sum(axis=1).astype(np.float32)

        ctx_cat = np.zeros((n, len(CTX_CATS)), dtype=np.int32)
        for j, name in enumerate(CTX_CATS):
            src = "ADES_mvt" if name == "ades" else name
            if name == "airport":
                ctx_cat[:, j] = encode_col(tgt["airport"], vocabs["airport"])
            elif src in tgt.columns:
                ctx_cat[:, j] = encode_col(tgt[src], vocabs[name])
            else:
                raise RuntimeError(f"missing context cat {name}")
        self.ctx_cat = ctx_cat

        # context numerics: clocks already on target frame; e20_pred; n_hist
        work = tgt
        if "n_hist" not in work.columns:
            work = work.with_columns(pl.Series("n_hist", n_hist))
        else:
            work = work.with_columns(pl.Series("n_hist", n_hist))
        if "e20_pred" not in work.columns:
            work = work.with_columns(pl.Series("e20_pred", self.e20))
        raw_ctx = _num_matrix(work, CTX_NUM_NAMES)
        # overwrite n_hist / e20 in case scaler expects them in the matrix
        raw_ctx[:, CTX_NUM_NAMES.index("n_hist")] = n_hist
        raw_ctx[:, CTX_NUM_NAMES.index("e20_pred")] = self.e20.astype(np.float64)
        self.ctx_num = ctx_scaler.transform(raw_ctx)
        if zero_e20_context:
            self.ctx_num[:, CTX_NUM_NAMES.index("e20_pred")] = 0.0

        self.tgt_rwy = ctx_cat[:, CTX_CATS.index("runway")]
        self.tgt_stand = ctx_cat[:, CTX_CATS.index("stand")]
        self.tgt_airline = ctx_cat[:, CTX_CATS.index("airline")]
        self.tgt_actype = ctx_cat[:, CTX_CATS.index("actype")]

    def leakage_sample(self, k: int = 2000, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        k = min(k, self.n)
        sel = rng.choice(self.n, size=k, replace=False)
        n_future = 0
        n_self = 0
        n_empty = 0
        for i in sel:
            m = self.mask[i]
            idx = self.seq_idx[i][m]
            if idx.size == 0:
                n_empty += 1
                continue
            if np.any(self.store.times[idx] >= self.tgt_time[i]):
                n_future += 1
            if self.pos[i] in idx:
                n_self += 1
        return {
            "n_checked": int(k),
            "n_future": int(n_future),
            "n_self": int(n_self),
            "n_empty": int(n_empty),
            "ok": n_future == 0 and n_self == 0,
        }

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        idx = self.seq_idx[i]
        mask = self.mask[i]
        seq_cat = self.store.cat[idx]
        seq_num = self.store.num[idx]
        dt = np.clip((self.tgt_time[i] - self.store.times[idx]).astype(np.float64), 0.0, 24 * 3600.0)
        dt = np.where(mask, dt, 0.0)
        rel = np.zeros((self.cfg.seq_len, len(REL_NUM_NAMES)), dtype=np.float32)
        rel[:, 0] = (dt / 3600.0).astype(np.float32)
        rel[:, 1] = np.log1p(dt).astype(np.float32) / 8.0
        rel[:, 2] = ((self.store.rwy[idx] == self.tgt_rwy[i]) & (self.store.rwy[idx] > 1) & mask).astype(np.float32)
        rel[:, 3] = ((self.store.stand[idx] == self.tgt_stand[i]) & (self.store.stand[idx] > 1) & mask).astype(np.float32)
        rel[:, 4] = ((self.store.airline[idx] == self.tgt_airline[i]) & (self.store.airline[idx] > 1) & mask).astype(np.float32)
        rel[:, 5] = ((self.store.actype[idx] == self.tgt_actype[i]) & (self.store.actype[idx] > 1) & mask).astype(np.float32)
        rel[:, 6] = (self.store.is_dep[idx].astype(np.bool_) & mask).astype(np.float32)
        return {
            "seq_cat": torch.from_numpy(seq_cat),
            "seq_num": torch.from_numpy(seq_num),
            "rel_num": torch.from_numpy(rel),
            "mask": torch.from_numpy(mask),
            "ctx_cat": torch.from_numpy(self.ctx_cat[i]),
            "ctx_num": torch.from_numpy(self.ctx_num[i]),
            "y": torch.tensor(self.y[i], dtype=torch.float32),
            "e20": torch.tensor(self.e20[i], dtype=torch.float32),
            "lirf_override": torch.tensor(self.lirf_override[i], dtype=torch.bool),
            "index": i,
        }


def collate(batch: list[dict]) -> dict:
    keys = ["seq_cat", "seq_num", "rel_num", "mask", "ctx_cat", "ctx_num", "y", "e20", "lirf_override"]
    out = {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}
    out["index"] = torch.tensor([b["index"] for b in batch], dtype=torch.long)
    return out
