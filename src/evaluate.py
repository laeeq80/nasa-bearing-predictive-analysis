# src/evaluate.py
"""
Evaluate a trained autoencoder on IMS processed signals.

Outputs:
 - <out-dir>/recon_errors.csv       : index, reconstruction_error, normalized_error
 - <out-dir>/recon_error_plot.png   : visualization of normalized error across file index
 - <out-dir>/summary.json           : small metadata summary (n_files, best_model, etc.)

Usage examples:
  python src/evaluate.py --processed-root ./data/processed --test 3rd_test --model-dir models --out-dir eval_results --batch-size 128

Notes:
 - This script expects `models/ae_best.pt` to exist (saved by model_train.py).
 - It also expects the folder `data/processed/<test>/` with meta.json and either signals.npy or signals_memmap.dat.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# try to import dataset/model definitions created during training
try:
    from model_train import IMSMemmapDataset, ConvAutoencoder
except Exception as e:
    raise ImportError(
        "Could not import IMSMemmapDataset / ConvAutoencoder from model_train.py. "
        "Make sure model_train.py is in src/ and contains these classes. "
        f"Original error: {e}"
    )

def evaluate(processed_root: Path, test: str, model_dir: Path, out_dir: Path, batch_size: int = 64, device: str = "cpu"):
    data_path = Path(processed_root) / test
    if not data_path.exists():
        raise FileNotFoundError(f"Processed folder not found: {data_path}")

    meta_path = data_path / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {data_path}. Run preprocess first.")

    meta = json.loads(meta_path.read_text())
    n_files = int(meta['n_files_found'])
    all_idx = np.arange(n_files)

    # load checkpoint
    ckpt_path = Path(model_dir) / "ae_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Run model_train.py first.")

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    mean = ckpt.get('mean', None)
    std = ckpt.get('std', None)
    downsample = int(ckpt.get('downsample', 16))
    # meta from checkpoint may also include preprocessing meta
    prep_meta = ckpt.get('meta', None)

    # build model: the ConvAutoencoder expects input_length = downsampled length
    sig_len = int(meta['signal_length'])
    if sig_len % downsample != 0:
        raise ValueError(f"Downsample factor {downsample} does not divide signal length {sig_len}")
    down_len = sig_len // downsample

    model = ConvAutoencoder(input_length=down_len)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # dataset + dataloader
    ds = IMSMemmapDataset(data_path, meta, all_idx, downsample_factor=downsample, mean=mean, std=std)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # device
    device = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")
    model.to(device)

    # compute reconstruction error (MSE per file)
    recon_errors = []
    with torch.no_grad():
        for batch in dl:
            batch = batch.to(device)              # shape (B, 1, L)
            out = model(batch)                    # same shape
            mse_per_sample = ((batch - out) ** 2).mean(dim=[1,2]).cpu().numpy()  # MSE per sample
            recon_errors.extend(list(mse_per_sample))

    recon_errors = np.array(recon_errors)
    # normalize to 0-1 for visualization
    e_min, e_max = float(recon_errors.min()), float(recon_errors.max())
    norm_err = (recon_errors - e_min) / (e_max - e_min + 1e-9)

    # save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "index": np.arange(len(recon_errors)),
        "reconstruction_error": recon_errors,
        "normalized_error": norm_err
    })
    csv_path = out_dir / "recon_errors.csv"
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved reconstruction errors to {csv_path}")

    # simple plot
    plt.figure(figsize=(12,4))
    plt.plot(df['index'], df['normalized_error'], label='normalized reconstruction error')
    # mark train/val/test split if meta has n_files info (assume training used first 60%)
    n = len(df)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    plt.axvline(train_end, color='orange', linestyle='--', linewidth=1, label='train_end (60%)')
    plt.axvline(val_end, color='red', linestyle='--', linewidth=1, label='val_end (80%)')
    plt.xlabel('file index')
    plt.ylabel('anomaly score (0-1)')
    plt.title('Reconstruction-based anomaly score')
    plt.legend()
    plot_path = out_dir / "recon_error_plot.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[INFO] Saved plot to {plot_path}")

    # summary
    summary = {
        "n_files": int(n_files),
        "n_scores": int(len(recon_errors)),
        "checkpoint": str(ckpt_path.resolve()),
        "csv": str(csv_path.resolve()),
        "plot": str(plot_path.resolve()),
        "downsample": downsample,
        "mean": mean,
        "std": std
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[INFO] Wrote summary.json to {out_dir}")

    print("[DONE] Evaluation finished successfully.")


def cli():
    parser = argparse.ArgumentParser(description="Evaluate trained AE model on processed IMS signals.")
    parser.add_argument("--processed-root", type=str, default="data/processed",
                        help="Processed data root (contains processed/<test>/)")
    parser.add_argument("--test", type=str, required=True,
                        help="Which processed test folder to evaluate (e.g. test1 or 3rd_test)")
    parser.add_argument("--model-dir", type=str, default="models", help="Folder containing ae_best.pt")
    parser.add_argument("--out-dir", type=str, default="results_eval", help="Folder to write evaluation outputs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model on (cpu or cuda:0)")
    args = parser.parse_args()
    evaluate(Path(args.processed_root), args.test, Path(args.model_dir), Path(args.out_dir), batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    cli()
