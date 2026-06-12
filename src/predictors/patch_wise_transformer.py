"""Skeleton Patch-wise Transformer model.

Honestly this isn't working well, here are two papers and repos 
I want to look more into to use

https://arxiv.org/pdf/2501.10448 (this includes weak data enriching) https://github.com/wangmeng-xpu/LiPFormer/blob/main/model/Lip.py

https://arxiv.org/abs/2211.14730 (this is just patch-wise transformer) https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/PatchTST_backbone.py

"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import logging

log = logging.getLogger(__name__)   

#configuration 
seq_len = 1440      # 1 day
patch_len = 60      # 1 hour
stride = 30     # 30 minutes
pred_horizon = 60  # predict 1 hour ahead
d_model = 128      # dimension of the token embeddings
n_heads = 4        # 4 attention heads
n_layers = 2       # 2 layers
ff_dim = 256       # 256 units in the feedforward network
dropout = 0.1      # 10% dropout
out_dim = 1        # 1 output dimension
use_cls_token = True

@dataclass
class PatchWiseTransformerConfig:
    """Configuration for PatchWiseTransformer."""

    input_dim: int = 1
    patch_len: int = patch_len
    stride: int = stride
    d_model: int = d_model  
    n_heads: int = n_heads
    n_layers: int = n_layers
    ff_dim: int = ff_dim
    dropout: float = dropout
    out_dim: int = out_dim
    use_cls_token: bool = use_cls_token


class PatchEmbedding(nn.Module):
    """Convert contiguous input patches to token embeddings."""

    def __init__(self, cfg: PatchWiseTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(cfg.patch_len * cfg.input_dim, cfg.d_model)

    def forward(self, x_patches: Tensor) -> Tensor:
        """Project flattened patches into d_model tokens.

        Args:
            x_patches: shape [batch, n_patches, patch_len * input_dim]
        Returns:
            Token embeddings: shape [batch, n_patches, d_model]
        """
        return self.proj(x_patches)


class PatchWiseTransformer(nn.Module):
    """Skeleton patch-wise Transformer for sequence modeling."""

    def __init__(self, cfg: PatchWiseTransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.patch_embed = PatchEmbedding(cfg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model)) if cfg.use_cls_token else None

        # Positional embedding is initialized lazily once n_patches is known.
        self.pos_embed: nn.Parameter | None = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.out_dim)

    def _extract_patches(self, x: Tensor) -> Tensor:
        """Split input into flattened patches.

        Expected input shape: [batch, seq_len, input_dim].
        Returns shape: [batch, n_patches, patch_len * input_dim].
        """
        bsz, seq_len, in_dim = x.shape
        if in_dim != self.cfg.input_dim:
            raise ValueError(f"input_dim mismatch: expected {self.cfg.input_dim}, got {in_dim}")

        if seq_len < self.cfg.patch_len:
            raise ValueError("seq_len must be >= patch_len")

        # x.unfold over time returns [batch, n_patches, input_dim, patch_len].
        # Permute to [batch, n_patches, patch_len, input_dim] before flattening
        # so each token contains one contiguous time patch across all features.
        patches = x.unfold(dimension=1, size=self.cfg.patch_len, step=self.cfg.stride)
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.contiguous().view(bsz, -1, self.cfg.patch_len * self.cfg.input_dim)
        return patches

    def _ensure_pos_embed(self, n_tokens: int, device: torch.device) -> None:
        """Create/resize positional embeddings if needed."""
        if self.pos_embed is None or self.pos_embed.shape[1] != n_tokens:
            pos = torch.zeros(1, n_tokens, self.cfg.d_model, device=device)
            nn.init.trunc_normal_(pos, std=0.02)
            self.pos_embed = nn.Parameter(pos)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            y: [batch, out_dim]
        """
        patches = self._extract_patches(x)
        tokens = self.patch_embed(patches)

        if self.cfg.use_cls_token:
            cls = self.cls_token.expand(tokens.size(0), -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)

        self._ensure_pos_embed(n_tokens=tokens.size(1), device=tokens.device)
        tokens = tokens + self.pos_embed

        h = self.encoder(tokens)
        h = self.norm(h)

        pooled = h[:, 0] if self.cfg.use_cls_token else h.mean(dim=1)
        return self.head(pooled)

class TimeSeriesWindowDataset(Dataset):
    """Simple windowed dataset skeleton.

    Expects:
      X: [num_samples, seq_len, input_dim]
      y: [num_samples, out_dim]
    """

    def __init__(self, X: Tensor, y: Tensor) -> None:
        if X.ndim != 3:
            raise ValueError("X must have shape [num_samples, seq_len, input_dim]")
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if X.size(0) != y.size(0):
            raise ValueError("X and y must have same num_samples")
        self.X = X.float()
        self.y = y.float()

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.X[idx], self.y[idx]


def build_dataloader(X: Tensor, y: Tensor, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """Build a DataLoader from window tensors."""
    ds = TimeSeriesWindowDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_train_test_tensors_from_csv(
    data_path: str | Path,
    seq_len: int,
    input_dim: int,
    train_ratio: float = 0.8,
    pred_horizon: int = 1,
    leakage_gap: int | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Load BTC candles, create rolling windows, and split chronologically.

    Features:
      [open, high, low, close, volume] (requires input_dim=5)
    Target:
      horizon-step log return from window end close.

    A gap is inserted between train and test windows so the final train
    window/target does not overlap heavily with the first test window.
    """
    if input_dim != 5:
        raise ValueError("This loader expects input_dim=5 for OHLCV features.")

    df = pd.read_csv(data_path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True, errors="coerce")
        df = df.sort_values("open_time").reset_index(drop=True)

    feature_cols = ["open", "high", "low", "close", "volume"]
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    values = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    closes = values[:, 3]

    num_rows = values.size(0)
    max_start = num_rows - seq_len - pred_horizon + 1
    if max_start <= 0:
        raise ValueError("Not enough rows for requested seq_len/pred_horizon.")

    X_list: list[Tensor] = []
    y_list: list[Tensor] = []
    for start in range(max_start):
        end = start + seq_len
        window = values[start:end]

        close_now = closes[end - 1]
        close_future = closes[end + pred_horizon - 1]
        target = torch.log(close_future / close_now).unsqueeze(0)

        X_list.append(window)
        y_list.append(target)

    X_all = torch.stack(X_list, dim=0)  # [N, seq_len, 5]
    y_all = torch.stack(y_list, dim=0)  # [N, 1]

    split_idx = int(len(X_all) * train_ratio)
    split_idx = max(1, min(split_idx, len(X_all) - 1))

    # Avoid leakage from near-identical overlapping windows around the split.
    # With seq_len=168 and pred_horizon=24, this removes a buffer of 192
    # candidate windows before the test set.
    if leakage_gap is None:
        leakage_gap = seq_len + pred_horizon
    train_end = max(1, split_idx - leakage_gap)

    X_train, X_test = X_all[:train_end], X_all[split_idx:]
    y_train, y_test = y_all[:train_end], y_all[split_idx:]

    print(
        f"windows: train={len(X_train)} test={len(X_test)} "
        f"gap={split_idx - train_end} seq_len={seq_len} pred_horizon={pred_horizon}"
    )

    # Fit normalization only on train to avoid leakage.
    feat_mean = X_train.mean(dim=(0, 1), keepdim=True)
    feat_std = X_train.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
    X_train = (X_train - feat_mean) / feat_std
    X_test = (X_test - feat_mean) / feat_std

    return X_train, y_train, X_test, y_test


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Single epoch training loop skeleton."""
    model.train()
    total_loss = 0.0

    for x, y in tqdm(dataloader, desc="Training", leave=False):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Simple evaluation loop on validation/test data."""
    model.eval()
    total_loss = 0.0

    for x, y in tqdm(dataloader, desc="Evaluating", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item()

    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def zero_return_baseline_loss(y: Tensor, criterion: nn.Module) -> float:
    """Baseline for log-return prediction: predict 0 return every time."""
    return criterion(torch.zeros_like(y), y).item()


if __name__ == "__main__":
    cfg = PatchWiseTransformerConfig(input_dim=5, out_dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PatchWiseTransformer(cfg).to(device)

    data_path = Path(__file__).resolve().parents[2] / "data" / "btc_minutes_100000.csv"

    X_train, y_train, X_test, y_test = load_train_test_tensors_from_csv(
        data_path=data_path,
        seq_len=seq_len,
        input_dim=cfg.input_dim,
        pred_horizon=pred_horizon,
    )
    train_loader = build_dataloader(X_train, y_train, batch_size=32, shuffle=True)
    test_loader = build_dataloader(X_test, y_test, batch_size=32, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    baseline_loss = zero_return_baseline_loss(y_test, criterion)
    print(f"zero_return_baseline_test_loss={baseline_loss:.6f}")

    print(f"Training {len(train_loader)} epochs")

    for epoch in tqdm(range(10), desc="Training", leave=False):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"epoch={epoch:02d} train_loss={train_loss:.6f} test_loss={test_loss:.6f}")

    out_path = Path(__file__).resolve().parent / "patch_wise_transformer.pth"
    torch.save(model.state_dict(), out_path)
    print(f"saved model to {out_path}")
