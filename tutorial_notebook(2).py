# -*- coding: utf-8 -*-
"""tutorial_notebook.py

# Equivariant SSL Tutorial Notebook
Code is based on https://github.com/SonyCSLParis/pesto-full
"""

from typing import Any, Dict, Sequence, Tuple, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from data import AudioDataModule, CQTModule, ToLogMagnitude

# ── Global Configuration ─────────────────────────────────────────────────────

# Data / CQT settings
HOP_DURATION        = 10.       # ms
FMIN                = 27.5      # Hz
BINS_PER_SEMITONE   = 3
N_BINS              = 88
CENTER_BINS         = True
BATCH_SIZE          = 256

# Training settings
MAX_EPOCHS          = 2
MAX_STEPS           = BINS_PER_SEMITONE * 11 // 2
MIN_STEPS           = -MAX_STEPS
T_MAX               = MAX_EPOCHS   # for CosineAnnealingLR

# Augmentation settings
MIN_SNR             = 0.1
MAX_SNR             = 2.
P_RANDNOISE         = 0.7
MIN_GAIN            = 0.5
MAX_GAIN            = 1.5
P_RANDGAIN          = 0.7

# Dataloader settings
NUM_WORKERS         = 2
PIN_MEMORY          = True
CACHE_DIR           = "./cache"

# Encoder settings
N_CHAN_INPUT        = 1
N_CHAN_LAYERS       = (40, 30, 30, 10, 3)
N_PREFILT_LAYERS    = 2
PREFILT_KERNEL_SIZE = 15
RESIDUAL            = False
N_BINS_RAW          = N_BINS * BINS_PER_SEMITONE - 1
N_BINS_IN           = N_BINS_RAW - MAX_STEPS + MIN_STEPS
OUTPUT_DIM          = 128 * BINS_PER_SEMITONE
A_LRELU             = 0.3
P_DROPOUT           = 0.2

# Optimizer settings
LR                  = 1e-4
WEIGHT_DECAY        = 0.

# Loss weights
LOSS_WEIGHTS        = dict(invariance=0., equivariance=1.0, shift_entropy=0.)
EMA_RATE            = 0.99

# Paths
AUDIO_FILES         = "train_files_vocadito_full_paths_small.csv"
CHECKPOINT_PATH     = "pesto_model_from_notebook.pt"

# ─────────────────────────────────────────────────────────────────────────────

"""## utils"""

def reduce_activations(activations: torch.Tensor, reduction: str = "alwa") -> torch.Tensor:
    device = activations.device
    num_bins = activations.size(1)
    bps, r = divmod(num_bins, 128)
    assert r == 0, "Activations should have output size 128*bins_per_semitone"

    all_pitches = torch.arange(num_bins, dtype=torch.float, device=device).div_(bps)

    if reduction == "alwa":
        center_bin = activations.argmax(dim=1, keepdim=True)
        window = torch.arange(1, 2 * bps, device=device) - bps
        indices = (window + center_bin).clip_(min=0, max=num_bins - 1)
        cropped_activations = activations.gather(1, indices)
        cropped_pitches = all_pitches.unsqueeze(0).expand_as(activations).gather(1, indices)
        return (cropped_activations * cropped_pitches).sum(dim=1) / cropped_activations.sum(dim=1)

    raise ValueError


def mid_to_hz(pitch: int):
    return 440 * 2 ** ((pitch - 69) / 12)


def generate_synth_data(pitch: int, num_harmonics: int = 5, duration=2, sr=16000):
    f0 = mid_to_hz(pitch)
    t = torch.arange(0, duration, 1/sr)
    harmonics = torch.stack([
        torch.cos(2 * torch.pi * k * f0 * t + torch.rand(()))
        for k in range(1, num_harmonics+1)
    ], dim=1)
    volume = torch.rand(num_harmonics)
    volume[0] = 1
    volume *= torch.randn(())
    audio = torch.sum(volume * harmonics, dim=1)
    return audio


"""## Transforms"""

class BatchRandomNoise(nn.Module):
    def __init__(self, min_snr: float = 0.0001, max_snr: float = 0.01, p: Optional[float] = None):
        super().__init__()
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        snr = torch.empty(batch_size, device=device).uniform_(self.min_snr, self.max_snr)
        mask = torch.rand_like(snr).le(self.p)
        snr[mask] = 0
        noise_std = snr * x.view(batch_size, -1).std(dim=-1)
        noise_std = noise_std.unsqueeze(-1).expand_as(x.view(batch_size, -1)).view_as(x)
        return x + noise_std * torch.randn_like(x)


class BatchRandomGain(nn.Module):
    def __init__(self, min_gain: float = 0.5, max_gain: float = 1.5, p: Optional[float] = None):
        super().__init__()
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        device = x.device
        vol = torch.empty(batch_size, device=device).uniform_(self.min_gain, self.max_gain)
        mask = torch.rand_like(vol).le(self.p)
        vol[mask] = 1
        vol = vol.unsqueeze(-1).expand_as(x.view(batch_size, -1)).view_as(x)
        return vol * x


def randint_sampling_fn(min_value, max_value):
    def sample_randint(*size, **kwargs):
        return torch.randint(min_value, max_value+1, size, **kwargs)
    return sample_randint


class PitchShiftCQT(nn.Module):
    def __init__(self, min_steps: int, max_steps: int):
        super().__init__()
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.sample_random_steps = randint_sampling_fn(min_steps, max_steps)
        self.lower_bin = self.max_steps

    def forward(self, spectrograms: torch.Tensor):
        batch_size, _, input_height = spectrograms.size()
        output_height = input_height - self.max_steps + self.min_steps
        assert output_height > 0

        n_steps = self.sample_random_steps(batch_size, device=spectrograms.device)
        x = spectrograms[..., self.lower_bin: self.lower_bin + output_height]
        xt = self.extract_bins(spectrograms, self.lower_bin - n_steps, output_height)
        return x, xt, n_steps

    def extract_bins(self, inputs: torch.Tensor, first_bin: torch.LongTensor, output_height: int):
        indices = first_bin.unsqueeze(-1) + torch.arange(output_height, device=inputs.device)
        dims = inputs.size(0), 1, output_height
        output_size = list(inputs.size())[:-1] + [output_height]
        indices = indices.view(*dims).expand(output_size)
        return inputs.gather(-1, indices)


"""## Losses"""

class PowerSeries(nn.Module):
    def __init__(self, value: float, power_min, power_max, tau: float = 1.):
        super().__init__()
        self.value = value
        self.tau = tau
        powers = torch.arange(power_min, power_max)
        self.register_buffer("weights", self.value ** powers, persistent=False)

    def forward(self, x1, x2, target, nlog_c1=None, nlog_c2=None):
        z1 = self.project(x1)
        z2 = self.project(x2)
        if nlog_c1 is not None:
            z1 = z1 * torch.exp(-nlog_c1)
        if nlog_c2 is not None:
            z2 = z2 * torch.exp(-nlog_c2)
        freq_ratios = self.value ** target.float()
        loss_12 = self._huber(z2 / z1 - freq_ratios).mean()
        loss_21 = self._huber(z1 / z2 - 1/freq_ratios).mean()
        return (loss_12 + loss_21) / 2

    def _huber(self, x):
        x = x.abs()
        return torch.where(x <= self.tau, x ** 2 / 2, self.tau ** 2 / 2 + self.tau * (x - self.tau))

    def project(self, x):
        return x.mv(self.weights)


class CrossEntropyLoss(nn.Module):
    def __init__(self, symmetric=False, detach_targets=False, backend=nn.CrossEntropyLoss()):
        super().__init__()
        self.symmetric = symmetric
        self.detach_targets = detach_targets
        self.backend = backend

    def forward(self, input, target):
        if self.symmetric:
            return (self.compute_loss(input, target) + self.compute_loss(target, input)) / 2
        return self.compute_loss(input, target)

    def compute_loss(self, input, target):
        return self.backend(input, target.detach() if self.detach_targets else target)


class ShiftCrossEntropy(nn.Module):
    def __init__(self, pad_length=5, criterion=CrossEntropyLoss()):
        super().__init__()
        self.criterion = criterion
        self.pad_length = pad_length

    def forward(self, x1, x2, target):
        x1 = F.pad(x1, (self.pad_length, self.pad_length))
        x2 = F.pad(x2, (2*self.pad_length, 2*self.pad_length))
        idx = target.unsqueeze(1) + torch.arange(x1.size(-1), device=target.device) + self.pad_length
        shift_x2 = torch.gather(x2, dim=1, index=idx)
        return self.criterion(x1, shift_x2)


"""## Loss Weighting"""

class GradientsLossWeighting:
    def __init__(self, weights: Mapping[str, float], ema_rate: float = 0.):
        self.weights = weights
        self.last_layer = None
        self.ema_rate = ema_rate
        self.grads = {k: 1-v for k, v in weights.items()}
        self.weights_tensor = None

    def setup(self, device):
        self.weights_tensor = torch.zeros(len(self.weights.keys()), device=device)

    def combine_losses(self, **losses):
        self.update_weights(losses)
        return sum(self.weights[k] * losses[k] for k in self.weights)

    def update_weights(self, losses):
        for i, (k, loss) in enumerate(losses.items()):
            if not loss.requires_grad:
                return
            grads = torch.autograd.grad(loss, self.last_layer, retain_graph=True)[0].norm().detach()
            old_grads = self.grads[k]
            if old_grads is not None:
                grads = self.ema_rate * old_grads + (1 - self.ema_rate) * grads
            self.grads[k] = grads
            self.weights_tensor[i] = grads
        self.weights_tensor = 1 - self.weights_tensor / self.weights_tensor.sum().clip(min=1e-7)
        for i, k in enumerate(losses.keys()):
            self.weights[k] = self.weights_tensor[i]


"""## Model"""

class ToeplitzLinear(nn.Conv1d):
    def __init__(self, in_features, out_features):
        super().__init__(
            in_channels=1,
            out_channels=1,
            kernel_size=in_features+out_features-1,
            padding=out_features-1,
            bias=False
        )

    def forward(self, input):
        return super().forward(input.unsqueeze(-2)).squeeze(-2)


class Resnet1d(nn.Module):
    def __init__(self,
                 n_chan_input=1,
                 n_chan_layers=(20, 20, 10, 1),
                 n_prefilt_layers=1,
                 prefilt_kernel_size=15,
                 residual=False,
                 n_bins_in=216,
                 output_dim=128,
                 a_lrelu=0.3,
                 p_dropout=0.2):
        super().__init__()

        activation_layer = partial(nn.LeakyReLU, negative_slope=a_lrelu)
        n_in = n_chan_input
        n_ch = n_chan_layers

        self.layernorm = nn.LayerNorm(normalized_shape=[n_in, n_bins_in])
        prefilt_padding = prefilt_kernel_size // 2
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_in, n_ch[0], prefilt_kernel_size, padding=prefilt_padding),
            activation_layer(),
            nn.Dropout(p=p_dropout)
        )
        self.n_prefilt_layers = n_prefilt_layers
        self.prefilt_layers = nn.ModuleList(*[
            nn.Sequential(
                nn.Conv1d(n_ch[0], n_ch[0], prefilt_kernel_size, padding=prefilt_padding),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            )
            for _ in range(n_prefilt_layers-1)
        ])
        self.residual = residual

        conv_layers = []
        for i in range(len(n_chan_layers)-1):
            conv_layers.extend([
                nn.Conv1d(n_ch[i], n_ch[i+1], kernel_size=1),
                activation_layer(),
                nn.Dropout(p=p_dropout)
            ])
        self.conv_layers = nn.Sequential(*conv_layers)

        self.flatten = nn.Flatten(start_dim=1)
        self.fc = ToeplitzLinear(n_bins_in * n_ch[-1], output_dim)
        self.final_norm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.layernorm(x)
        x = self.conv1(x)
        for p in range(self.n_prefilt_layers - 1):
            prefilt_layer = self.prefilt_layers[p]
            x = x + prefilt_layer(x) if self.residual else prefilt_layer(x)
        x = self.conv_layers(x)
        x = self.flatten(x)
        return self.final_norm(self.fc(x))


class PESTO(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 equiv_loss_fn=None,
                 sce_loss_fn=None,
                 inv_loss_fn=None,
                 pitch_shift_kwargs=None,
                 transforms: Sequence[nn.Module] | None = None):
        super().__init__()
        self.encoder = encoder
        self.equiv_loss_fn = equiv_loss_fn
        self.sce_loss_fn = sce_loss_fn
        self.inv_loss_fn = inv_loss_fn
        self.pitch_shift = PitchShiftCQT(**(pitch_shift_kwargs or {}))
        self.transforms = nn.Sequential(*transforms) if transforms is not None else nn.Identity()
        self.register_buffer('shift', torch.zeros((), dtype=torch.float), persistent=True)

    def forward(self, x: torch.Tensor, shift: bool = True) -> torch.Tensor:
        x, *_ = self.pitch_shift(x)
        preds = reduce_activations(self.encoder(x))
        if shift:
            preds.sub_(self.shift)
        return preds

    def training_step(self, x, loss_weighting):
        x, xt, n_steps = self.pitch_shift(x)
        xa = self.transforms(x.clone())
        xt = self.transforms(xt)

        y  = self.encoder(x)
        ya = self.encoder(xa)
        yt = self.encoder(xt)

        inv_loss          = self.inv_loss_fn(y, ya)
        shift_entropy_loss = self.sce_loss_fn(ya, yt, n_steps)
        equiv_loss        = self.equiv_loss_fn(ya, yt, n_steps)

        total_loss = loss_weighting.combine_losses(
            invariance=inv_loss, shift_entropy=shift_entropy_loss, equivariance=equiv_loss
        )
        return total_loss, dict(invariance=inv_loss, equivariance=equiv_loss,
                                shift_entropy=shift_entropy_loss, loss=total_loss)

    def estimate_shift(self, datamodule) -> None:
        labels = torch.arange(60, 72)
        sr = 16000
        batch = []
        for p in labels:
            audio = generate_synth_data(p, sr=sr)
            cqt = datamodule.cqt(audio, sr)                        # (time, 1, freq_bins, 2)
            log_cqt = ToLogMagnitude()(torch.view_as_complex(cqt.float()))  # (time, 1, freq_bins)
            batch.append(log_cqt[0])

        x = torch.stack(batch, dim=0).to(next(self.parameters()).device)
        preds = self.forward(x, shift=False)

        diff = preds - labels.to(x.device)
        shift, std = diff.median(), diff.std()
        self.shift.fill_(shift)


"""## Train"""

device = "cuda" if torch.cuda.is_available() else "cpu"

datamodule = AudioDataModule(
    audio_files=AUDIO_FILES,
    hop_duration=HOP_DURATION,
    fmin=FMIN,
    bins_per_semitone=BINS_PER_SEMITONE,
    n_bins=N_BINS_RAW,
    center_bins=CENTER_BINS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    cache_dir=CACHE_DIR,
)
datamodule.prepare_data()

encoder = Resnet1d(
    n_chan_input=N_CHAN_INPUT,
    n_chan_layers=list(N_CHAN_LAYERS),
    n_prefilt_layers=N_PREFILT_LAYERS,
    prefilt_kernel_size=PREFILT_KERNEL_SIZE,
    residual=RESIDUAL,
    n_bins_in=N_BINS_IN,
    output_dim=OUTPUT_DIM,
    a_lrelu=A_LRELU,
    p_dropout=P_DROPOUT
)

equiv_loss = PowerSeries(value=2**(1/36), power_min=1-OUTPUT_DIM, power_max=1, tau=2**(1/6)-1)
inv_loss   = CrossEntropyLoss(symmetric=True, detach_targets=True)
sce_loss   = ShiftCrossEntropy(pad_length=MAX_STEPS, criterion=CrossEntropyLoss(symmetric=True, detach_targets=True))

transforms = [
    BatchRandomNoise(min_snr=MIN_SNR, max_snr=MAX_SNR, p=P_RANDNOISE),
    BatchRandomGain(min_gain=MIN_GAIN, max_gain=MAX_GAIN, p=P_RANDGAIN),
]

model = PESTO(
    encoder=encoder,
    equiv_loss_fn=equiv_loss,
    inv_loss_fn=inv_loss,
    sce_loss_fn=sce_loss,
    transforms=transforms,
    pitch_shift_kwargs=dict(min_steps=MIN_STEPS, max_steps=MAX_STEPS),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)

loss_weighting = GradientsLossWeighting(weights=LOSS_WEIGHTS, ema_rate=EMA_RATE)
loss_weighting.last_layer = model.encoder.fc.weight
loss_weighting.setup(device)

# ── Training loop ─────────────────────────────────────────────────────────────
train_loader = datamodule.train_dataloader()

for epoch in range(MAX_EPOCHS):
    model.train()
    model.estimate_shift(datamodule)

    for batch_idx, (x,) in enumerate(train_loader):       # TensorDataset yields single-element tuples
        x = x.to(device)

        optimizer.zero_grad()
        total_loss, loss_dict = model.training_step(x, loss_weighting)
        total_loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Step {batch_idx} | " +
                  " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items()))

    scheduler.step()

torch.save({"state_dict": model.state_dict()}, CHECKPOINT_PATH)


"""## Inference"""

audio_path = "/content/AClassicEducation_NightOwl_STEM_01.RESYN.wav"

ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["state_dict"])
model.eval()

waveform, sr = torchaudio.load(audio_path)
audio_mono = waveform.mean(dim=0)

hop_length = int(HOP_DURATION * sr / 1000 + 0.5)
cqt_module = CQTModule(
    sr=sr,
    hop_length=hop_length,
    **datamodule.cqt_kwargs,
)
log_mag = ToLogMagnitude()

with torch.no_grad():
    cqt = cqt_module(audio_mono)                                        # (time, 1, freq_bins, 2)
    cqt = log_mag(torch.view_as_complex(cqt.float()))                  # (time, 1, freq_bins)

pitcher = model.pitch_shift
all_preds, all_acts = [], []
BATCH = 256

with torch.no_grad():
    for start in range(0, len(cqt), BATCH):
        x = cqt[start:start+BATCH]                                     # already log-mag float
        x_shifted, _, _ = pitcher(x)
        acts = model.encoder(x_shifted)
        preds = reduce_activations(acts, reduction="alwa") - model.shift
        all_preds.append(preds)
        all_acts.append(acts)

pitches     = torch.cat(all_preds).numpy()
activations = torch.cat(all_acts).numpy()
times       = np.arange(len(pitches)) * hop_length / sr

# ── Display ───────────────────────────────────────────────────────────────────
cqt_display = cqt.squeeze().numpy()      # (T, freq_bins) — already log-mag

midi_fmin  = 12 * np.log2(27.5 / 440) + 69
bps        = datamodule.cqt_kwargs["bins_per_semitone"]
pitch_bins = (pitches - midi_fmin) * bps
valid      = (pitch_bins >= 0) & (pitch_bins < cqt_display.shape[1])

fig, ax = plt.subplots(figsize=(14, 5))
ax.imshow(cqt_display.T, aspect="auto", origin="lower", cmap="magma",
          extent=[times[0], times[-1], 0, cqt_display.shape[1]])
ax.plot(times, pitch_bins, c="cyan", linewidth=1, alpha=0.9, label="Predicted pitch")
ax.set_xlabel("Time (s)")
ax.set_ylabel("MIDI pitch")
ax.set_title(f"PESTO Predictions — {audio_path.split('/')[-1]}")

notes      = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
midi_ticks = np.arange(int(pitches[valid].min()), int(pitches[valid].max()) + 1, 4)
ax.set_yticks((midi_ticks - midi_fmin) * bps)
ax.set_yticklabels([f"{notes[int(m)%12]}{int(m)//12-1}" for m in midi_ticks], fontsize=8)
ax.legend(loc="upper right")

plt.tight_layout()
plt.savefig("predictions.png", dpi=150, bbox_inches="tight")
plt.show()


"""## Visualise Embeddings"""

embeddings_list = []

def hook_fn(module, input, output):
    embeddings_list.append(input[0].detach().cpu())

hook = model.encoder.final_norm.register_forward_hook(hook_fn)

with torch.no_grad():
    for start in range(0, len(cqt), BATCH):
        x = cqt[start:start+BATCH]
        x_shifted, _, _ = pitcher(x)
        model.encoder(x_shifted)

hook.remove()

embeddings = torch.cat(embeddings_list).numpy()

MAX_POINTS = 5000
if len(embeddings) > MAX_POINTS:
    idx          = np.random.choice(len(embeddings), MAX_POINTS, replace=False)
    idx.sort()
    emb_sample   = embeddings[idx]
    pitch_sample = pitches[idx]
else:
    emb_sample   = embeddings
    pitch_sample = pitches

try:
    import umap
    reducer     = umap.UMAP(n_components=2, random_state=42)
    method_name = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    reducer     = TSNE(n_components=2, random_state=42, perplexity=30)
    method_name = "t-SNE"

print(f"Running {method_name} on {len(emb_sample)} points...")
emb_2d = reducer.fit_transform(emb_sample)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pitch_sample, cmap="plasma", s=4, alpha=0.7)
plt.colorbar(sc, ax=axes[0], label="MIDI pitch")
axes[0].set_title(f"{method_name} — coloured by pitch")
axes[0].set_xlabel(f"{method_name} 1")
axes[0].set_ylabel(f"{method_name} 2")

notes       = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
pitch_class = pitch_sample % 12
sc2 = axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=pitch_class, cmap="hsv", vmin=0, vmax=12, s=4, alpha=0.7)
cbar = plt.colorbar(sc2, ax=axes[1], ticks=np.arange(0.5, 12))
cbar.ax.set_yticklabels(notes, fontsize=8)
axes[1].set_title(f"{method_name} — coloured by pitch class")
axes[1].set_xlabel(f"{method_name} 1")
axes[1].set_ylabel(f"{method_name} 2")

plt.suptitle(f"Learned encoder embeddings — {audio_path.split('/')[-1]}", fontsize=11)
plt.tight_layout()
plt.savefig("embeddings.png", dpi=150, bbox_inches="tight")
plt.show()