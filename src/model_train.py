#model_train.py

# src/model_train.py
"""
Train a simple 1D autoencoder on preprocessed IMS bearing signals.

Design goals / notes (human-readable):
- Supports both numpy stacked files (signals.npy) and numpy memmap (signals_memmap.dat).
- To keep memory and model size manageable, we downsample each per-file signal by a small factor
  (default 16). IMS signals are typically 20480 samples per file -> downsampled to 1280.
- Training split = first 60% (assumed healthy), validation = next 20%, test = final 20% (optional).
- The script computes mean/std from the training split (streaming/chunked if memmap) and normalizes.
- Uses a small convolutional autoencoder on the downsampled 1D signals.
- Saves the best model (lowest val loss) to disk.
- Simple, robust, easy-to-run for demonstration / portfolio purposes.

Usage (example):
python src/model_train.py --processed-root ./data/processed --test test3 --use-memmap --epochs 30 --batch-size 64

"""

import json
from pathlib import Path
import argparse
import numpy as np
import math
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

# ---------------------------
# Dataset wrapper
# ---------------------------

class IMSMemmapDataset(Dataset):
    """
    Dataset that reads either a stacked signals.npy array or a memmap file.
    It returns downsampled and normalized 1D signals as torch.float32 tensors.
    """
    def __init__(self, data_path: Path, meta: dict, indices,
                 downsample_factor: int = 16, mean=None, std=None):
        """
        data_path: path to folder with signals (signals.npy or signals_memmap.dat)
        meta: dictionary loaded from meta.json (contains signal_length, n_files)
        indices: list or array of file indices to use (subset)
        downsample_factor: integer factor to reduce temporal length (must divide sig_len)
        mean, std: scalar normalization values computed on training set (if None, no norm)
        """
        self.data_path = Path(data_path)
        self.meta = meta
        self.indices = np.array(indices, dtype=int)
        self.down = int(downsample_factor)
        self.mean = float(mean) if mean is not None else None
        self.std = float(std) if std is not None else None

        # find actual signals storage
        signals_npy = self.data_path / "signals.npy"
        memmap_file = self.data_path / "signals_memmap.dat"
        self.sig_len = int(meta['signal_length'])
        # ensure downsample divides length
        if self.sig_len % self.down != 0:
            raise ValueError(f"Downsample factor {self.down} does not divide signal length {self.sig_len}")

        if signals_npy.exists():
            # load into read-only memory-mapped array to keep API consistent
            self.storage = np.load(signals_npy, mmap_mode='r')
            self.mode = 'npy'
        elif memmap_file.exists():
            # need to open memmap with proper shape
            n_files = int(meta['n_files_found'])
            self.storage = np.memmap(memmap_file, dtype='float32', mode='r', shape=(n_files, self.sig_len))
            self.mode = 'memmap'
        else:
            raise FileNotFoundError("No signals.npy or signals_memmap.dat found in " + str(self.data_path))

        # downsampled length
        self.down_len = self.sig_len // self.down

    def __len__(self):
        return len(self.indices)

    def _downsample(self, arr):
        """
        Simple non-overlapping block mean downsampling.
        arr shape: (sig_len,)
        returns shape: (down_len,)
        """
        # reshape to (down_len, down) and mean along axis=1
        reshaped = arr.reshape(self.down_len, self.down)
        down = reshaped.mean(axis=1)
        return down.astype(np.float32)

    def __getitem__(self, idx):
        file_index = int(self.indices[idx])
        arr = self.storage[file_index]  # memmap or mmap numpy
        down = self._downsample(arr)
        if self.mean is not None and self.std is not None:
            down = (down - self.mean) / (self.std + 1e-9)
        # return as (channels, length) for Conv1d: channels=1
        return torch.from_numpy(down).unsqueeze(0)  # shape (1, down_len)


# ---------------------------
# Small Conv Autoencoder
# ---------------------------

class ConvAutoencoder(nn.Module):
    """
    Small convolutional autoencoder for 1D signals.
    Keeps the model compact so it trains quickly on a laptop.
    Architecture: Conv -> Conv -> Conv (bottleneck) -> ConvTranspose -> ConvTranspose -> ConvTranspose
    """
    def __init__(self, input_length):
        super().__init__()
        # encoder
        self.enc = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),   # length // 2
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),  # length // 4
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),  # length // 8
            nn.ReLU()
        )
        # compute encoded length for decoder sizing
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_length)
            enc_out = self.enc(dummy)
            self.encoded_len = enc_out.shape[-1]  # temporal length after convs

        # decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=9, stride=2, padding=4, output_padding=1),
            # final output should be same length as input
        )

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        # If output length mismatches, trim or pad (safeguard)
        if out.shape[-1] > x.shape[-1]:
            out = out[..., :x.shape[-1]]
        elif out.shape[-1] < x.shape[-1]:
            pad = x.shape[-1] - out.shape[-1]
            out = torch.nn.functional.pad(out, (0, pad))
        return out


# ---------------------------
# Utilities
# ---------------------------

def compute_normalization_from_memmap(data_path: Path, meta: dict, train_end: int, downsample_factor: int = 16, chunk=512):
    """
    Compute global mean and std over training set in a streaming fashion.
    We compute these on the DOWNsampled signals to match model input.
    chunk: number of files to read at once from memmap (balances IO & RAM)
    """
    sig_len = int(meta['signal_length'])
    if sig_len % downsample_factor != 0:
        raise ValueError("downsample_factor must divide signal length")

    down_len = sig_len // downsample_factor
    # open storage
    signals_npy = data_path / "signals.npy"
    memmap_file = data_path / "signals_memmap.dat"
    if signals_npy.exists():
        storage = np.load(signals_npy, mmap_mode='r')
    elif memmap_file.exists():
        n_files = int(meta['n_files_found'])
        storage = np.memmap(memmap_file, dtype='float32', mode='r', shape=(n_files, sig_len))
    else:
        raise FileNotFoundError("No signals.npy or signals_memmap.dat found")

    # incremental mean/std: compute mean and mean of squares
    total_count = 0
    mean_acc = np.zeros(down_len, dtype=np.float64)
    m2_acc = np.zeros(down_len, dtype=np.float64)

    for start in range(0, train_end, chunk):
        end = min(train_end, start + chunk)
        batch = storage[start:end]  # shape (b, sig_len)
        # downsample batch: reshape (b, down_len, down) and mean over last axis
        b = batch.shape[0]
        reshaped = batch.reshape(b, down_len, downsample_factor)
        down = reshaped.mean(axis=2)  # shape (b, down_len)
        # update running mean/std per time-step (we will then average to scalar mean/std)
        # flatten across time to compute scalar mean/std across all elements
        flat = down.reshape(-1)
        if total_count == 0:
            mean_acc_scalar = flat.mean()
            m2_acc_scalar = ((flat - mean_acc_scalar) ** 2).sum()
            total_count = flat.size
            mean_global = mean_acc_scalar
            m2_global = m2_acc_scalar
        else:
            # combine existing and new
            new_count = flat.size
            new_mean = flat.mean()
            new_m2 = ((flat - new_mean) ** 2).sum()
            # Welford combination
            delta = new_mean - mean_global
            combined_count = total_count + new_count
            mean_global = (mean_global * total_count + new_mean * new_count) / combined_count
            m2_global = m2_global + new_m2 + delta * delta * total_count * new_count / combined_count
            total_count = combined_count

    variance = m2_global / (total_count - 1) if total_count > 1 else 0.0
    std = math.sqrt(variance)
    mean = float(mean_global)
    return mean, std


# ---------------------------
# Training loop
# ---------------------------

def train(args):
    processed_root = Path(args.processed_root)
    test_folder = args.test
    data_path = processed_root / test_folder
    if not data_path.exists():
        raise FileNotFoundError(f"Processed folder not found: {data_path}")

    meta_path = data_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json missing in {data_path}. Run preprocess first.")
    meta = json.loads(meta_path.read_text())

    n_files = int(meta['n_files_found'])
    # split indices
    train_end = int(0.6 * n_files)
    val_end = int(0.8 * n_files)
    all_indices = np.arange(n_files)
    train_idx = all_indices[:train_end]
    val_idx = all_indices[train_end:val_end]
    test_idx = all_indices[val_end:]

    # compute normalization (on downsampled signals) from training data
    print("[INFO] Computing normalization (mean/std) from training split...")
    mean, std = compute_normalization_from_memmap(data_path, meta, train_end, downsample_factor=args.downsample)
    print(f"[INFO] mean={mean:.6f}, std={std:.6f}")

    # create datasets and dataloaders
    train_ds = IMSMemmapDataset(data_path, meta, train_idx, downsample_factor=args.downsample, mean=mean, std=std)
    val_ds = IMSMemmapDataset(data_path, meta, val_idx, downsample_factor=args.downsample, mean=mean, std=std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"[INFO] Using device: {device}")

    # model with input_length = downsampled length
    input_length = train_ds.down_len
    model = ConvAutoencoder(input_length=input_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    best_val = float('inf')
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "ae_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}", unit="batch") as pbar:
            for batch in train_loader:
                batch = batch.to(device)  # shape (B, 1, L)
                optimizer.zero_grad()
                out = model(batch)
                loss = criterion(out, batch)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * batch.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.6f}"})
                pbar.update(1)
        epoch_loss = running_loss / (len(train_loader.dataset))
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss = val_loss / (len(val_loader.dataset) if len(val_loader.dataset) > 0 else 1.0)
        print(f"[EPOCH {epoch}] train_loss={epoch_loss:.6f}, val_loss={val_loss:.6f}")

        # save best
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'mean': mean,
                'std': std,
                'downsample': args.downsample,
                'meta': meta
            }, best_path)
            print(f"[INFO] Saved best model to {best_path} (val_loss={best_val:.6f})")

    print("[DONE] Training complete.")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(description="Train autoencoder on preprocessed IMS signals.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Root folder where processed/<test>/ exists")
    parser.add_argument("--test", type=str, default="test3", help="Which processed test folder to use")
    parser.add_argument("--model-dir", type=str, default="models", help="Where to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--downsample", type=int, default=16, help="Downsample factor (must divide signal length)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use-memmap", action="store_true", help="(ignored) kept for compatibility with preprocess flag")
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
