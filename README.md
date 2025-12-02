# **NASA Bearing Anomaly Detection Using Autoencoder**

This repository implements an end-to-end anomaly detection pipeline for the NASA/IMS bearing vibration dataset. It covers data preprocessing, feature extraction, deep learning model training, evaluation, and visualization of anomaly progression. The project is designed as a compact but realistic example of an industrial AI workflow for machine condition monitoring.

The solution uses a lightweight 1D convolutional autoencoder trained exclusively on healthy bearing signals. Reconstruction error is then used as an unsupervised anomaly score to detect early degradation and failure.

**Repository Structure**
```text
project/
│
├── src/
│   ├── preprocess.py        # Preprocesses raw NASA IMS data into signals + features
│   ├── model_train.py       # Trains 1D Conv Autoencoder and saves best checkpoint
│   ├── evaluate.py          # Computes reconstruction-based anomaly scores
│
├── data/
│   ├── raw/                 # (Git-ignored) place NASA test1/test2/test3 folders here
│   └── processed/           # Preprocessed signals, features, and metadata
│
├── models/
│   └── ae_best.pt           # Trained autoencoder model (best validation loss)
│
├── demo_notebook/
│   └── demo.ipynb           # End-to-end demonstration notebook
│
├── requirements.txt
└── README.md
```

Raw data and large intermediate files are intentionally excluded via .gitignore.

**Dataset**

The project uses the public **NASA IMS bearing dataset**, a widely used benchmark for studying vibration-based condition monitoring.
Each test set contains thousands of high-frequency vibration recordings captured during run-to-failure experiments.

Expected directory structure under data/raw/:

```
data/raw/
    1st_test/
    2nd_test/
    3rd_test/
```
Each test folder contains thousands of ASCII vibration files.

**Installation**

Create and activate a virtual environment, then install dependencies:
```
pip install -r requirements.txt
```
The project uses PyTorch (CPU or CUDA builds) together with NumPy, SciPy, Pandas, and Matplotlib.

**Step 1: Preprocessing**

`preprocess.py` converts raw ASCII vibration files into:

`signals.npy` or `signals_memmap.dat`

`features.csv` (per-file statistical features)

`meta.json` (signal length, file count, etc.)

Example:
```
python src/preprocess.py --data-root ./data/raw --test test3 --use-memmap
```
Outputs appear under:
```
data/processed/3rd_test/
```
Memmap mode is recommended for large datasets.

**Step 2: Model Training**

`model_train.py` trains a compact 1D convolutional autoencoder on the downsampled, normalized signals. Only the best validation model is saved.

Example:
```
python src/model_train.py --processed-root ./data/processed --test 3rd_test --use-memmap --epochs 30 --batch-size 64
```
This produces:
```
models/ae_best.pt
```
The checkpoint contains:

model weights

optimizer state

downsample factor

training statistics (mean, std)

preprocessing metadata

**Step 3: Evaluation**

`evaluate.py` loads the best checkpoint and computes reconstruction error for each sample in the dataset.

Example:
```
python src/evaluate.py --processed-root ./data/processed --test 3rd_test --model-dir models --out-dir eval_results
```
This produces:

`recon_errors.csv`

`recon_error_plot.png`

`summary.json`

The reconstruction error curve typically shows a clear rise corresponding to bearing degradation and eventual failure, validating the model's anomaly detection capability.

**Results Interpretation**

The autoencoder is trained on healthy data (first 60 percent). Reconstruction error remains low during healthy operation and increases sharply as the bearing deteriorates. The final region of the curve shows the failure phase, where the autoencoder cannot reconstruct the abnormal vibration pattern.

This behavior is consistent with expected degradation progression in the IMS Test 3 dataset.

**Notebook Demonstration**

A `demo_notebook/demo.ipynb` notebook is included to visualize:

raw and downsampled signals

feature distributions

the learned autoencoder architecture

reconstruction-based anomaly trends

failure region detection

This notebook serves as a concise walk-through for reviewers or hiring committees evaluating the project.

**Notes**

This project is structured for clarity and reproducibility rather than maximum model accuracy.

Only the best model checkpoint is stored to keep the repository clean.

The preprocessing pipeline is designed to handle tens of thousands of high-frequency files efficiently.


If you want, I can also generate a concise “Project Summary” section suitable for your CV or RISE application, or review your GitHub repository before you make it public.
