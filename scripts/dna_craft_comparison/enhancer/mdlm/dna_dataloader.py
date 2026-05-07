"""DNA-specific dataloader + tokenizer for MDLM DiMamba pretraining.

Designed to drop into the MDLM training loop in place of `dataloader.get_dataloaders`
and `dataloader.get_tokenizer`. The tokenizer maps the 4 DNA bases plus the
diffusion specials (PAD / MASK) onto tokens 0-5; the dataloader streams the
parquet shards produced by `data/curate_encode_ccre.py`.

Token map (matches the curate script + `mdlm_wrapper.py`):
    A=0, C=1, G=2, T=3, PAD=4, MASK=5

The MDLM diffusion module reads `tokenizer.vocab_size` and a couple of special
token attrs (`mask_token_id`, `pad_token_id`); everything else just needs to
exist for HF compatibility.
"""
from __future__ import annotations

import typing
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import transformers
from torch.utils.data import DataLoader, Dataset

ACGT_BASES = ["A", "C", "G", "T"]
A, C, G, T, PAD, MASK = 0, 1, 2, 3, 4, 5
VOCAB = ["A", "C", "G", "T", "[PAD]", "[MASK]"]


class DNATokenizer(transformers.PreTrainedTokenizer):
    """6-token tokenizer: ACGT + PAD + MASK. Drop-in for MDLM."""

    def __init__(
        self,
        pad_token: str = "[PAD]",
        mask_token: str = "[MASK]",
        bos_token: str = "[PAD]",
        eos_token: str = "[PAD]",
        unk_token: str = "[PAD]",
        sep_token: str = "[PAD]",
        cls_token: str = "[PAD]",
        **kwargs,
    ):
        self.characters = ACGT_BASES
        self._vocab_str_to_int = {tok: i for i, tok in enumerate(VOCAB)}
        self._vocab_int_to_str = {i: tok for tok, i in self._vocab_str_to_int.items()}
        super().__init__(
            pad_token=pad_token,
            mask_token=mask_token,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str, **_kw) -> typing.List[str]:
        return list(text.upper())

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[PAD]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str.get(index, "[PAD]")

    def convert_tokens_to_string(self, tokens) -> str:
        return "".join(t for t in tokens if t in ACGT_BASES)

    def get_vocab(self):
        return self._vocab_str_to_int


# -----------------------------------------------------------------------------
# Parquet-backed dataset
# -----------------------------------------------------------------------------
class CCREParquetDataset(Dataset):
    """Loads `tokens` lists from parquet rows into LongTensors.

    Each row contains a `tokens` field (list[int], length 350) and a
    `region_id` string. We pre-load the full parquet table into a numpy
    array (shape N x L) at __init__ time — at the curated 90/5/5 split
    sizes (≈3 M / 0.16 M / 0.16 M rows × 350 bp = ≈1 GB train), this fits
    in RAM and avoids per-batch IO overhead.
    """

    def __init__(self, parquet_path: Path):
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(parquet_path)
        table = pq.read_table(str(path), columns=["tokens"])
        rows = table.column("tokens").to_pylist()
        self.tokens = np.asarray(rows, dtype=np.int64)
        if self.tokens.ndim != 2:
            raise ValueError(
                f"expected (N, L) tokens, got shape {self.tokens.shape}")

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> dict:
        ids = torch.from_numpy(self.tokens[idx])
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
    }


# -----------------------------------------------------------------------------
# MDLM-compatible factory functions
# -----------------------------------------------------------------------------
def get_tokenizer(_config):
    """Drop-in replacement for `dataloader.get_tokenizer`."""
    return DNATokenizer()


def _build_loader(parquet_path: Path, batch_size: int, num_workers: int,
                  pin_memory: bool, shuffle: bool, distributed: bool,
                  seed: int) -> DataLoader:
    """Build a vanilla DataLoader.

    We deliberately do NOT pre-attach a `DistributedSampler` here, even when
    `distributed=True`. mdlm's `get_dataloaders` is called from `main._train`
    BEFORE `trainer.fit()` runs, so torch.distributed isn't initialized yet
    when this function fires — constructing a `DistributedSampler` at that
    point raises `Default process group has not been initialized`. Instead we
    rely on PyTorch Lightning's `use_distributed_sampler=True` (the PL 2.x
    default) to wrap our DataLoader inside a DistributedSampler automatically
    once the Trainer's DDP strategy has set up the process group.
    """
    del distributed, seed  # Lightning handles the DDP-aware sampler
    ds = CCREParquetDataset(parquet_path)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_collate,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )


def get_dataloaders(config, tokenizer=None, skip_train: bool = False,
                    valid_seed: int | None = None):
    """Drop-in replacement for `dataloader.get_dataloaders`.

    Reads `config.data.parquet_dir`, expects `train.parquet` / `val.parquet`.
    Both DataLoaders are tagged with `.tokenizer` so MDLM's `_train` can read
    `valid_ds.tokenizer` (matching the original interface).
    """
    pq_dir = Path(config.data.parquet_dir)
    train_path = pq_dir / "train.parquet"
    val_path = pq_dir / "val.parquet"
    distributed = (
        config.trainer.devices > 1 or config.trainer.num_nodes > 1)

    seed = config.seed if valid_seed is None else valid_seed
    tk = tokenizer or DNATokenizer()
    val_loader = _build_loader(
        val_path, config.loader.eval_batch_size,
        config.loader.num_workers, config.loader.pin_memory,
        shuffle=False, distributed=distributed, seed=seed,
    )
    val_loader.tokenizer = tk
    if skip_train:
        return None, val_loader
    train_loader = _build_loader(
        train_path, config.loader.batch_size,
        config.loader.num_workers, config.loader.pin_memory,
        shuffle=True, distributed=distributed, seed=seed,
    )
    train_loader.tokenizer = tk
    return train_loader, val_loader


__all__ = [
    "DNATokenizer",
    "CCREParquetDataset",
    "get_tokenizer",
    "get_dataloaders",
]
