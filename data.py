# -*- coding: utf-8 -*-
"""data.py — Simplified CQT data loading for equivariant SSL pitch tracking."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from tqdm import tqdm

from nnAudio.features.cqt import CQT

log = logging.getLogger(__name__)


# ── CQT Module ────────────────────────────────────────────────────────────────

class CQTModule(nn.Module):
    def __init__(
            self,
            sr: int = 22050,
            hop_length: int = 512,
            fmin: float = 32.7,
            fmax: float | None = None,
            bins_per_semitone: int = 1,
            n_bins: int = 84,
            center_bins: bool = True
    ):
        super().__init__()

        if center_bins:
            fmin = fmin / 2 ** ((bins_per_semitone - 1) / (24 * bins_per_semitone))

        self.cqt_kernel = CQT(
            sr=sr,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            n_bins=n_bins,
            bins_per_octave=12 * bins_per_semitone,
            output_format="Complex",
            verbose=False
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: mono waveform, shape (samples,)
        Returns:
            CQT, shape (time, 1, freq_bins)
        """
        # (1, freq_bins, time, 2) -> (time, 1, freq_bins, 2)
        return self.cqt_kernel(audio).squeeze(0).permute(1, 0, 2).unsqueeze(1)


# ── Log Magnitude ─────────────────────────────────────────────────────────────

class ToLogMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = torch.finfo(torch.float32).eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) == 2:
            x = torch.view_as_complex(x)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return x.abs().clamp(min=self.eps).log10().mul(20)


# ── Data Module ───────────────────────────────────────────────────────────────

class AudioDataModule:
    def __init__(
            self,
            audio_files: str,
            hop_duration: float = 10.,
            fmin: float = 27.5,
            fmax: float | None = None,
            bins_per_semitone: int = 1,
            n_bins: int = 84,
            center_bins: bool = True,
            batch_size: int = 256,
            num_workers: int = 2,
            pin_memory: bool = False,
            cache_dir: str = "./cache",
    ):
        self.audio_files = Path(audio_files)
        self.hop_duration = hop_duration
        self.cqt_kwargs = dict(
            fmin=fmin,
            fmax=fmax,
            bins_per_semitone=bins_per_semitone,
            n_bins=n_bins,
            center_bins=center_bins,
        )
        self.dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.cache_dir = Path(cache_dir)

        # expose for downstream use (e.g. inference, shift estimation)
        self.cqt_kwargs = self.cqt_kwargs
        self._cqt_sr: int | None = None
        self._cqt_module: CQTModule | None = None
        self._log_mag = ToLogMagnitude()

        self.dataset: torch.utils.data.TensorDataset | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def prepare_data(self) -> None:
        tensor = self._load_or_compute()
        self.dataset = torch.utils.data.TensorDataset(tensor)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        assert self.dataset is not None, "Call prepare_data() first."
        sampler = torch.utils.data.RandomSampler(self.dataset)
        return torch.utils.data.DataLoader(self.dataset, sampler=sampler, **self.dl_kwargs)

    def cqt(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """Compute CQT for a single mono waveform. Reuses kernel if sr unchanged."""
        if sr != self._cqt_sr:
            self._cqt_sr = sr
            hop_length = int(self.hop_duration * sr / 1000 + 0.5)
            self._cqt_module = CQTModule(sr=sr, hop_length=hop_length, **self.cqt_kwargs)
        return self._cqt_module(audio)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _cache_path(self) -> Path:
        key = json.dumps({
            "audio_files": str(self.audio_files),
            "hop_duration": self.hop_duration,
            **self.cqt_kwargs,
        }, sort_keys=True)
        hash_id = hashlib.sha256(key.encode()).hexdigest()[:8]
        return self.cache_dir / f"cqt_{hash_id}.pt"

    def _load_or_compute(self) -> torch.Tensor:
        cache = self._cache_path()
        if cache.exists():
            log.info(f"Loading CQT cache from {cache}")
            return torch.load(cache, weights_only=True)

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tensor = self._precompute_cqt()
        torch.save(tensor, cache)
        log.info(f"Saved CQT cache to {cache}")
        return tensor

    def _precompute_cqt(self) -> torch.Tensor:
        audio_files = self.audio_files.read_text().splitlines()
        data_dir = self.audio_files.parent

        chunks = []
        for fname in tqdm(audio_files, desc="Precomputing CQT", leave=False):
            fname = fname.strip()
            if not fname:
                continue
            waveform, sr = torchaudio.load(data_dir / fname)
            cqt = self.cqt(waveform.mean(dim=0), sr)           # (time, 1, freq_bins, 2)
            log_cqt = self._log_mag(torch.view_as_complex(cqt.float()))  # (time, 1, freq_bins)
            chunks.append(log_cqt.cpu())

        return torch.cat(chunks)                                 # (total_frames, 1, freq_bins)