# src/preprocess.py
"""
Preprocessing script for NASA IMS bearing dataset.

What it does (high level):
- Reads ASCII vibration files from a given test-folder (e.g. test1/test2/test3).
- Extracts a single channel (or falls back to first column).
- Ensures every signal has the same length (trim or zero-pad).
- Computes a small set of per-file statistical features.
- Writes features.csv and either a stacked signals.npy or a memory-mapped file
  (signals_memmap.dat) depending on the --use-memmap flag.
- Writes a meta.json summarizing what was produced.

Designed to be robust for large folders (thousands of files).
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import argparse
import json
import datetime
import sys

# -------------------------
# Helper functions
# -------------------------

def read_ascii_file(path):
    """
    Load a single IMS ASCII file. Returns a numpy array.
    IMS files typically are one value per line (20,480 rows) but some mirrors
    may have columns. np.loadtxt is simple and reliable here.
    """
    return np.loadtxt(path, dtype=np.float32)


def compute_features(signal):
    """
    Compute a small set of descriptive features for a 1D signal.
    Keep features simple and interpretable (mean, std, rms, skew, kurtosis, FFT peak).
    """
    feats = {}
    feats['mean'] = float(np.mean(signal))
    feats['std'] = float(np.std(signal))
    feats['rms'] = float(np.sqrt(np.mean(np.square(signal))))
    # scipy.stats provides robust skew/kurtosis functions
    feats['skew'] = float(stats.skew(signal))
    feats['kurtosis'] = float(stats.kurtosis(signal))
    # quick spectral feature: index of peak magnitude in rFFT
    fft = np.fft.rfft(signal)
    mags = np.abs(fft)
    feats['peak_freq_idx'] = int(np.argmax(mags))
    return feats


def ensure_length(signal, target_len):
    """
    Make sure the signal has length `target_len`.
    - If longer: trim to target_len (simple and deterministic).
    - If shorter: pad with zeros at the end.
    """
    if signal.shape[0] == target_len:
        return signal
    if signal.shape[0] > target_len:
        return signal[:target_len]
    # shorter -> pad with zeros
    padded = np.zeros(target_len, dtype=np.float32)
    padded[: signal.shape[0]] = signal
    return padded


# -------------------------
# Main processing routine
# -------------------------

def process_test_set(test_path: Path, out_dir: Path, channel_index: int = 0, use_memmap: bool = False, progress_every: int = 500):
    """
    Process all files in test_path and write outputs to out_dir.
    Outputs:
      - features.csv   (per-file features + filename + index)
      - signals.npy OR signals_memmap.dat (memmap)
      - meta.json (summary)
    """
    test_path = Path(test_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect file list (only files, sorted deterministically)
    files = sorted([p for p in test_path.iterdir() if p.is_file()])
    n_files = len(files)
    if n_files == 0:
        print(f"[ERROR] No files found in {test_path}")
        return

    # peek first valid file to determine signal length and dimensionality
    first_signal = None
    first_idx = None
    for i, f in enumerate(files):
        try:
            arr = read_ascii_file(f)
            # pick channel if multi-column; otherwise arr is 1D
            if arr.ndim == 2:
                if arr.shape[1] > channel_index:
                    first_signal = arr[:, channel_index]
                else:
                    first_signal = arr[:, 0]
            else:
                first_signal = arr
            first_idx = i
            break
        except Exception as e:
            # skip unreadable files but log
            print(f"[WARN] Skipping unreadable file {f.name}: {e}")
            continue

    if first_signal is None:
        print(f"[ERROR] Could not read any files from {test_path}")
        return

    sig_len = int(first_signal.shape[0])
    print(f"[INFO] Found {n_files} files. Signal length (per file) inferred as {sig_len} samples.")

    # Prepare outputs
    features = []
    skipped_files = []
    meta = {
        "source_path": str(test_path.resolve()),
        "n_files_found": n_files,
        "signal_length": sig_len,
        "channel_index": channel_index,
        "use_memmap": bool(use_memmap),
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }

    if use_memmap:
        # create memmap file of shape (n_files, sig_len)
        mmap_path = out_dir / "signals_memmap.dat"
        # 'w+' mode creates file and allows reading/writing
        mm = np.memmap(mmap_path, dtype='float32', mode='w+', shape=(n_files, sig_len))
        print(f"[INFO] Created memmap at {mmap_path}")
    else:
        collected = []  # careful with memory for large datasets

    # Process files one by one
    processed_count = 0
    for idx, f in enumerate(files):
        try:
            arr = read_ascii_file(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f.name} (skipping): {e}")
            skipped_files.append(str(f.name))
            continue

        # handle 2D vs 1D file content
        if arr.ndim == 2:
            if arr.shape[1] > channel_index:
                signal = arr[:, channel_index]
            else:
                signal = arr[:, 0]
        else:
            signal = arr

        # ensure consistent length
        signal = ensure_length(signal.astype(np.float32), sig_len)

        # compute features for this file
        feats = compute_features(signal)
        feats['file'] = str(f.name)
        feats['index'] = int(idx)
        features.append(feats)

        # store signal
        if use_memmap:
            mm[idx, :] = signal
        else:
            collected.append(signal)

        processed_count += 1
        # progress log
        if (processed_count % progress_every) == 0:
            print(f"[INFO] Processed {processed_count}/{n_files} files...")

    # finalize outputs
    features_df = pd.DataFrame(features).sort_values('index')
    features_csv = out_dir / "features.csv"
    features_df.to_csv(features_csv, index=False)
    print(f"[INFO] Wrote features to {features_csv} ({len(features_df)} rows)")

    if use_memmap:
        # flush memmap to disk by deleting reference (important)
        del mm
        meta['signals_memmap'] = str(mmap_path.resolve())
        print(f"[INFO] Memmap flushed: {mmap_path}")
    else:
        # stack collected signals and save
        stacked = np.stack(collected, axis=0)  # shape (N, sig_len)
        signals_npy = out_dir / "signals.npy"
        np.save(signals_npy, stacked)
        meta['signals_npy'] = str(signals_npy.resolve())
        print(f"[INFO] Saved stacked signals to {signals_npy} with shape {stacked.shape}")

    # save meta data and skipped files
    meta['n_processed'] = processed_count
    meta['n_skipped'] = len(skipped_files)
    if skipped_files:
        meta['skipped_files'] = skipped_files
    meta_path = out_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[INFO] Wrote meta data to {meta_path}")

    print("[DONE] preprocessing complete.")


# -------------------------
# Script entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Preprocess IMS bearing ASCII files into features + signals (npy or memmap).")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Root folder containing test subfolders (e.g. data/raw)")
    parser.add_argument("--out-root", type=str, default="data/processed",
                        help="Output root folder where processed/<test>/ will be written")
    parser.add_argument("--test", type=str, default="test1",
                        help="Test folder name under data-root to process (e.g. test1)")
    parser.add_argument("--channel", type=int, default=0,
                        help="Channel index to extract (0-based). If file has single column, channel is ignored.")
    parser.add_argument("--use-memmap", action="store_true",
                        help="If set, write signals to a numpy memmap file instead of signals.npy (saves RAM).")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="How often to print progress (files).")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    test_path = data_root / args.test
    out_root = Path(args.out_root) / args.test

    if not test_path.exists():
        print(f"[ERROR] Test path does not exist: {test_path}", file=sys.stderr)
        sys.exit(1)

    process_test_set(test_path=test_path,
                     out_dir=out_root,
                     channel_index=args.channel,
                     use_memmap=args.use_memmap,
                     progress_every=args.progress_every)


if __name__ == "__main__":
    main()
