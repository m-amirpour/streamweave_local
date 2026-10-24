# ================================ CONFIG ====================================
SAMPLE_SIZE = 20000
SEEDS = list(range(10))
EPOCHS_BPU = 12
EPOCHS_KD = 5
STREAM_BS = 256
MEMORY_MAX = 4096
MAX_ENSEMBLE = 6  # must be >= HDT_K + SPU capacity
BATCH_INTERVAL = 2048
BPU_REFRESH_EPOCHS = 2
SPU_ONLINE_LR = 1e-3
SPU_WARMUP_EPOCHS = 5

TRAIN_FRAC = 0.50
CAL_FRAC = 0.10  # total calibration allocation
CAL_FIT_FRAC = 0.05  # fit CU parameters only
CAL_THRESH_FRAC = 0.05  # tune decision threshold only
ADAPT_FRAC = 0.10
STREAM_FRAC = 0.20
TEST_FRAC = 0.10

DATASETS_TO_RUN = ["Covertype", "Skin", "NSL-KDD"]
BPU_MODELS_TO_RUN = ["MLP", "DeepAE", "CNN1D"]
SPU_MODELS_TO_RUN = ["MLP", "GRU", "LightAE"]
TRANSFERS_TO_RUN = ["None", "StandardKD", "RKD", "EnsembleDistill"]
CUS_TO_RUN = ["HDT", "SimpleAvg", "ConfWeight", "Bayesian", "TempScale"]

PHASE1_FIXED_BPU = "DeepAE"
PHASE1_FIXED_SPU = "MLP"
PHASE2_FIXED_BPU = "DeepAE"
PHASE2_FIXED_SPU = "MLP"

ENABLE_DRIFT = True
DRIFT_POS_RATES = [0.20, 0.50, 0.80]
DRIFT_FEATURE_SHIFT = 0.50
DRIFT_FEATURE_SCALE = 1.00
DRIFT_SEGMENTS = 3

DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
RANDOM_STATE = 0
NUM_WORKERS = 0

# =============================== IMPORTS ====================================
import os
import copy
import gc
import json
import math
import time
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    average_precision_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Local-only paths. Override with environment variables if desired.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = str(
    Path(os.environ.get("STREAMWEAVE_DATA_DIR", BASE_DIR / "data"))
    .expanduser()
    .resolve()
)
SAVE_DIR = str(
    Path(
        os.environ.get(
            "STREAMWEAVE_RESULTS_DIR", BASE_DIR / "results" / "streamweave_v15"
        )
    )
    .expanduser()
    .resolve()
)


# ============================ LOCAL OUTPUT SETUP ============================
def prepare_save_dir():
    """Create and verify the local results directory; no cloud mounting."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    probe = os.path.join(SAVE_DIR, ".streamweave_write_test")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok\n")
        os.remove(probe)
    except OSError as e:
        raise RuntimeError(f"Results directory is not writable: {SAVE_DIR}: {e}") from e


prepare_save_dir()
print("Runtime: local Python")
print(f"Device: {DEVICE}")
print(f"Data/cache directory: {DATA_DIR}")
print(f"Results directory: {SAVE_DIR}")


# =============================== UTILITIES ==================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sigmoid_np(x):
    x = np.clip(x, -40, 40)
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics(y_true, p, threshold=0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p).reshape(-1)
    pred = (p >= threshold).astype(int)
    try:
        roc_auc = float(roc_auc_score(y_true, p))
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = float(average_precision_score(y_true, p))
    except ValueError:
        pr_auc = float("nan")
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "logloss": float(log_loss(y_true, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def tune_threshold(y, p) -> float:
    """Tune threshold on CALIBRATION data only."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p).reshape(-1)
    best_t, best_f = 0.5, -1.0
    for t in np.linspace(0.10, 0.90, 81):
        f = f1_score(y, (p >= t).astype(int), zero_division=0)
        if f > best_f:
            best_f, best_t = float(f), float(t)
    return best_t


def safe_auc_like_score(y, p):
    # Kept deliberately simple; the pipeline's primary metrics are accuracy/F1.
    return binary_metrics(y, p, 0.5)


def append_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    new = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    # Stable deduplication across reruns.
    subset = [
        c
        for c in [
            "dataset",
            "seed",
            "bpu",
            "spu",
            "transfer",
            "cu",
            "drift",
            "ablation",
            "baseline",
            "protocol",
        ]
        if c in out.columns
    ]
    if subset:
        out = out.drop_duplicates(subset=subset, keep="last")
    out.to_csv(path, index=False)


# ============================== DATA SPLITS =================================
@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal: np.ndarray
    y_cal: np.ndarray
    X_adapt: np.ndarray
    y_adapt: np.ndarray
    X_stream: np.ndarray
    y_stream: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: Any = None
    feature_names: Optional[List[str]] = None
    # X_cal/y_cal are aliases for the threshold-calibration partition.
    X_cal_fit: Optional[np.ndarray] = None
    y_cal_fit: Optional[np.ndarray] = None


def _subsample(X, y, n, seed=0):
    """Uniformly subsample rows without changing the dataset's class prior."""
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    if n is None or n >= len(y):
        return X, y
    if n <= 0:
        raise ValueError("sample size must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    rng.shuffle(idx)
    return X[idx], y[idx]


def _to_binary(y, positive=None):
    """Convert labels to binary without data-dependent class semantics."""
    y = np.asarray(y)
    vals = np.unique(y)
    if positive is not None:
        return (y == positive).astype(np.int64)
    if len(vals) != 2:
        raise ValueError("A multiclass dataset requires an explicit binary mapping.")
    return (y == vals[-1]).astype(np.int64)


def load_covertype(seed=0, sample_size=SAMPLE_SIZE):
    from sklearn.datasets import fetch_covtype

    ds = fetch_covtype()
    X = np.asarray(ds.data, dtype=np.float32)
    # StreamWeave/previous HDBS-KD protocol: classes 1 and 2 are normal;
    # classes 3--7 are anomalous. Do not infer this mapping from class counts.
    y = (np.asarray(ds.target) >= 3).astype(np.int64)
    return _subsample(X, y, sample_size, seed)


def load_skin(seed=0, sample_size=SAMPLE_SIZE):
    # Skin Segmentation (UCI 00229), not HAM10000. This loader accepts a local
    # CSV if supplied, otherwise downloads the UCI CSV when pandas can do so.
    local_candidates = [
        os.path.join(DATA_DIR, "Skin_NonSkin.txt"),
        os.path.join(DATA_DIR, "skin.csv"),
        "skin.csv",
        "Skin_NonSkin.txt",
    ]
    for path in local_candidates:
        if os.path.exists(path):
            if Path(path).suffix.lower() == ".txt":
                df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
            else:
                df = pd.read_csv(path, sep=None, engine="python")
            break
    else:
        # Download the UCI Skin Segmentation dataset when it is not cached locally.
        raw = pd.read_csv(
            "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt",
            sep="\t",
            header=None,
        )
        X = raw.iloc[:, :-1].to_numpy(np.float32)
        y = raw.iloc[:, -1].to_numpy()
        return _subsample(X, _to_binary(y), sample_size, seed)
    if df.shape[1] < 2:
        raise ValueError("Skin dataset must contain features and a label column.")
    y = _to_binary(df.iloc[:, -1].to_numpy())
    X = pd.get_dummies(df.iloc[:, :-1], dummy_na=True).to_numpy(np.float32)
    return _subsample(X, y, sample_size, seed)


def load_nsl_kdd(seed=0, sample_size=SAMPLE_SIZE):
    """Load NSL-KDD without encoding categories before the split.

    Categorical encoding is fitted on TRAIN only in split_and_preprocess().
    """
    url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    cols = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
        "difficulty",
    ]
    df = pd.read_csv(url, header=None, names=cols)
    y = (df["label"].astype(str).str.lower() != "normal").astype(np.int64).to_numpy()
    X = df.drop(columns=["label", "difficulty"])
    X, y = _subsample_dataframe(X, y, sample_size, seed)
    return X.reset_index(drop=True), y


def _subsample_dataframe(X, y, n, seed=0):
    """Uniformly subsample dataframe rows, preserving natural class prevalence."""
    y = np.asarray(y).astype(np.int64)
    if n is None or n >= len(y):
        return X.copy(), y
    if n <= 0:
        raise ValueError("sample size must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n, replace=False)
    rng.shuffle(idx)
    return X.iloc[idx].reset_index(drop=True), y[idx]


def load_raw_dataset(name, seed=0):
    if name == "Covertype":
        return load_covertype(seed)
    if name == "Skin":
        return load_skin(seed)
    if name == "NSL-KDD":
        return load_nsl_kdd(seed)
    raise ValueError(f"Unknown dataset: {name}")


def split_and_preprocess(X, y, seed=0) -> DataSplit:
    """Split first; fit every preprocessing transform on TRAIN only."""
    y = np.asarray(y).astype(np.int64)
    fracs = np.asarray(
        [TRAIN_FRAC, CAL_FRAC, ADAPT_FRAC, STREAM_FRAC, TEST_FRAC], dtype=float
    )
    if not np.isclose(fracs.sum(), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fracs.sum():.6f}")
    if not np.isclose(CAL_FIT_FRAC + CAL_THRESH_FRAC, CAL_FRAC):
        raise ValueError("CAL_FIT_FRAC + CAL_THRESH_FRAC must equal CAL_FRAC")
    if min(CAL_FIT_FRAC, CAL_THRESH_FRAC) <= 0:
        raise ValueError("Both calibration partitions must be positive")

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=1.0 - TRAIN_FRAC, stratify=y, random_state=seed
    )
    remaining = 1.0 - TRAIN_FRAC
    cal_rel = CAL_FRAC / remaining
    X_cal, X_tmp2, y_cal, y_tmp2 = train_test_split(
        X_tmp, y_tmp, test_size=1.0 - cal_rel, stratify=y_tmp, random_state=seed + 17
    )
    # Separate CU fitting from decision-threshold tuning. The two partitions
    # are disjoint and together preserve the configured 10% calibration budget.
    thresh_rel = CAL_THRESH_FRAC / CAL_FRAC
    X_cal_fit, X_cal_thresh, y_cal_fit, y_cal_thresh = train_test_split(
        X_cal, y_cal, test_size=thresh_rel, stratify=y_cal, random_state=seed + 19
    )
    remaining2 = ADAPT_FRAC + STREAM_FRAC + TEST_FRAC
    adapt_rel = ADAPT_FRAC / remaining2
    X_adapt, X_tmp3, y_adapt, y_tmp3 = train_test_split(
        X_tmp2,
        y_tmp2,
        test_size=1.0 - adapt_rel,
        stratify=y_tmp2,
        random_state=seed + 23,
    )
    remaining3 = STREAM_FRAC + TEST_FRAC
    stream_rel = STREAM_FRAC / remaining3
    X_stream, X_test, y_stream, y_test = train_test_split(
        X_tmp3,
        y_tmp3,
        test_size=1.0 - stream_rel,
        stratify=y_tmp3,
        random_state=seed + 29,
    )

    if isinstance(X_tr, pd.DataFrame):
        cat_cols = X_tr.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in X_tr.columns if c not in cat_cols]
        # Numeric imputation uses TRAIN statistics only.
        train_num = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
        medians = train_num.median()

        def prep_frame(df):
            num = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(medians)
            if cat_cols:
                cat = df[cat_cols].fillna("__MISSING__").astype(str)
                return pd.concat([num, cat], axis=1)
            return num

        X_tr2, X_cal_fit2, X_cal_thresh2, X_adapt2, X_stream2, X_test2 = map(
            prep_frame, [X_tr, X_cal_fit, X_cal_thresh, X_adapt, X_stream, X_test]
        )
        # Fit OHE on TRAIN only; unknown categories in later splits are ignored.
        if cat_cols:
            enc = OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, dtype=np.float32
            )
            enc.fit(X_tr2[cat_cols])

            def transform(df):
                num = df[num_cols].to_numpy(np.float32)
                cat = enc.transform(df[cat_cols])
                return np.hstack([num, cat]).astype(np.float32)

            A = [
                transform(z)
                for z in [
                    X_tr2,
                    X_cal_fit2,
                    X_cal_thresh2,
                    X_adapt2,
                    X_stream2,
                    X_test2,
                ]
            ]
        else:
            A = [
                z.to_numpy(np.float32)
                for z in [
                    X_tr2,
                    X_cal_fit2,
                    X_cal_thresh2,
                    X_adapt2,
                    X_stream2,
                    X_test2,
                ]
            ]
    else:
        A = [
            np.asarray(z, dtype=np.float32)
            for z in [X_tr, X_cal_fit, X_cal_thresh, X_adapt, X_stream, X_test]
        ]

    scaler = StandardScaler()
    X_tr, X_cal_fit, X_cal, X_adapt, X_stream, X_test = [
        scaler.fit_transform(A[0]).astype(np.float32),
        scaler.transform(A[1]).astype(np.float32),
        scaler.transform(A[2]).astype(np.float32),
        scaler.transform(A[3]).astype(np.float32),
        scaler.transform(A[4]).astype(np.float32),
        scaler.transform(A[5]).astype(np.float32),
    ]
    return DataSplit(
        X_train=X_tr,
        y_train=y_tr,
        X_cal=X_cal,
        y_cal=y_cal_thresh,
        X_adapt=X_adapt,
        y_adapt=y_adapt,
        X_stream=X_stream,
        y_stream=y_stream,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        X_cal_fit=X_cal_fit,
        y_cal_fit=y_cal_fit,
    )


# ============================== DRIFT =======================================
def inject_drift(
    X, y, target_rates=None, feature_shift=0.5, feature_scale=1.0, segments=3, seed=0
):
    """Create a synthetic stream with fixed total length.

    Each segment targets a positive-class rate. If the requested rate cannot be
    represented without replacement, sampling is performed with replacement for
    the minority class. This keeps the segment length exact and makes the actual
    rate equal to the requested rate up to rounding.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)
    n = len(y)
    rng = np.random.default_rng(seed)
    if target_rates is None:
        target_rates = [0.2, 0.5, 0.8]
    target_rates = list(target_rates)
    if len(target_rates) != segments:
        target_rates = (target_rates * segments)[:segments]

    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return X.copy(), y.copy()

    parts_x, parts_y = [], []
    bounds = np.linspace(0, n, segments + 1, dtype=int)
    for s in range(segments):
        m = bounds[s + 1] - bounds[s]
        n_pos = int(round(m * float(target_rates[s])))
        n_pos = min(max(n_pos, 0), m)
        n_neg = m - n_pos
        ip = (
            rng.choice(pos, n_pos, replace=n_pos > len(pos))
            if n_pos
            else np.empty(0, int)
        )
        ineg = (
            rng.choice(neg, n_neg, replace=n_neg > len(neg))
            if n_neg
            else np.empty(0, int)
        )
        idx = np.concatenate([ip, ineg])
        rng.shuffle(idx)
        xs = X[idx].copy()
        ys = y[idx].copy()
        # Segment-dependent feature perturbation. The first segment is unchanged.
        if s > 0:
            xs = xs * (1.0 + feature_scale * 0.05 * s)
            xs += rng.normal(0.0, feature_shift * 0.05 * s, size=xs.shape).astype(
                np.float32
            )
        parts_x.append(xs)
        parts_y.append(ys)
    return np.concatenate(parts_x), np.concatenate(parts_y)


# =============================== MODELS =====================================
class BinaryHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc = nn.Linear(d, 1)

    def forward(self, z):
        return self.fc(z).squeeze(-1)


class BPU_MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.head = BinaryHead(64)

    def forward_features(self, x):
        return self.net(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


class BPU_DeepAE(nn.Module):
    def __init__(self, d):
        super().__init__()
        h = min(128, max(32, d * 2))
        z = min(64, max(16, d))
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, z), nn.ReLU())
        self.head = BinaryHead(z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))

    def forward_features(self, x):
        return self.enc(x)

    def forward(self, x):
        return self.head(self.forward_features(x))

    def reconstruction(self, x):
        return self.dec(self.forward_features(x))


class BPU_CNN1D(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = BinaryHead(32)

    def forward_features(self, x):
        return self.conv(x.unsqueeze(1)).squeeze(-1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class BPU_LSTM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.head = BinaryHead(32)

    def forward_features(self, x):
        return self.lstm(x.unsqueeze(-1))[0][:, -1]

    def forward(self, x):
        return self.head(self.forward_features(x))


class BPU_Transformer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Linear(1, 32)
        enc = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, batch_first=True, dropout=0.1
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=1)
        self.head = BinaryHead(32)

    def forward_features(self, x):
        z = self.proj(x.unsqueeze(-1))
        z = self.enc(z)
        return z.mean(dim=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class SPU_MLP(BPU_MLP):
    pass


class SPU_GRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(1, 32, batch_first=True)
        self.head = BinaryHead(32)

    def forward_features(self, x):
        return self.gru(x.unsqueeze(-1))[0][:, -1]

    def forward(self, x):
        return self.head(self.forward_features(x))


class SPU_TCN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = BinaryHead(32)

    def forward_features(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)

    def forward(self, x):
        return self.head(self.forward_features(x))


class SPU_LightAE(BPU_DeepAE):
    pass


BPU_FACTORY = {
    "MLP": BPU_MLP,
    "DeepAE": BPU_DeepAE,
    "CNN1D": BPU_CNN1D,
    "LSTM": BPU_LSTM,
    "Transformer": BPU_Transformer,
}
SPU_FACTORY = {"MLP": SPU_MLP, "GRU": SPU_GRU, "TCN": SPU_TCN, "LightAE": SPU_LightAE}


def make_model(kind, d, bpu=True):
    return (BPU_FACTORY if bpu else SPU_FACTORY)[kind](d).to(DEVICE)


def reset_model(model):
    def _reset(m):
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
        if isinstance(m, nn.BatchNorm1d):
            m.reset_running_stats()

    model.apply(_reset)
    return model


# ============================= TRAINING =====================================
def tensor_batches(X, y, batch_size, shuffle, device_transfer=True):
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.float32)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )


def train_classifier(
    model,
    X,
    y,
    epochs=EPOCHS_BPU,
    lr=1e-3,
    batch_size=128,
    reconstruction=False,
    seed=0,
):
    set_seed(seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    loader = tensor_batches(X, y, batch_size, True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = bce(logits, yb)
            if reconstruction and hasattr(model, "reconstruction"):
                xr = model.reconstruction(xb)
                loss = loss + 0.10 * mse(xr, xb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    return model


def safe_train_bpu(kind, X, y, epochs=EPOCHS_BPU, seed=0, bpu=True):
    last_error = None
    for attempt in range(3):
        try:
            set_seed(seed + attempt * 1000)
            model = make_model(kind, X.shape[1], bpu=bpu)
            model = train_classifier(
                model,
                X,
                y,
                epochs=epochs,
                seed=seed + attempt * 1000,
                reconstruction=(kind in ["DeepAE", "LightAE"]),
            )
            with torch.no_grad():
                for p in model.parameters():
                    if not torch.isfinite(p).all():
                        raise FloatingPointError("Non-finite model parameter")
            return model
        except Exception as e:
            last_error = e
            cleanup()
    raise RuntimeError(f"Training failed for {kind}: {last_error}")


@torch.no_grad()
def predict_proba(model, X, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(
            DEVICE
        )
        out.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty(0, dtype=np.float32)


@torch.no_grad()
def extract_features(model, X, batch_size=512):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(np.asarray(X[i : i + batch_size], dtype=np.float32)).to(
            DEVICE
        )
        if hasattr(model, "forward_features"):
            z = model.forward_features(xb)
        else:
            z = xb
        out.append(z.detach().cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 1), dtype=np.float32)


# ============================ KNOWLEDGE TRANSFER ============================
def transfer_none(teacher, student, X, y, seed=0):
    # The student was already trained from scratch on TRAIN before transfer.
    # "None" means no additional transfer/fine-tuning stage.
    return student


def transfer_standard_kd(teacher, student, X, y, seed=0, T=2.0, alpha=0.5):
    set_seed(seed)
    teacher.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    loader = tensor_batches(X, y, 128, True)
    for _ in range(EPOCHS_KD):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                tlog = teacher(xb)
                tp = torch.sigmoid(tlog / T)
            slog = student(xb)
            sp = torch.sigmoid(slog / T)
            kd = F.binary_cross_entropy(sp, tp) * (T * T)
            hard = bce(slog, yb)
            loss = alpha * kd + (1 - alpha) * hard
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
    return student


def transfer_rkd(teacher, student, X, y, seed=0):
    """Relational KD using pairwise distances (feature widths may differ)."""
    set_seed(seed)
    teacher.eval()
    student.train()
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    batch_size = min(128, max(1, len(X)))

    for _ in range(EPOCHS_KD):
        for xb, yb in tensor_batches(X, y, batch_size, True):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                tz = F.normalize(teacher.forward_features(xb), dim=1)
                td = torch.cdist(tz, tz, p=2)
            sz = F.normalize(student.forward_features(xb), dim=1)
            sd = torch.cdist(sz, sz, p=2)
            mask = ~torch.eye(len(xb), dtype=torch.bool, device=DEVICE)
            tdv, sdv = td[mask], sd[mask]
            if tdv.numel():
                tdv = tdv / (tdv.mean().detach() + 1e-6)
                sdv = sdv / (sdv.mean().detach() + 1e-6)
                loss_rkd = F.smooth_l1_loss(sdv, tdv)
            else:
                loss_rkd = torch.zeros((), device=DEVICE)
            loss = bce(student(xb), yb) + 0.5 * loss_rkd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
            opt.step()
    return student


def transfer_born_again(teacher, student, X, y, seed=0):
    # Born-again style: supervised student + teacher soft targets, then the
    # resulting student becomes the transferred model.
    return transfer_standard_kd(teacher, student, X, y, seed=seed, T=2.0, alpha=0.35)


def transfer_fitnets(teacher, student, X, y, seed=0):
    set_seed(seed)
    teacher.eval()
    student.train()
    with torch.no_grad():
        tz = extract_features(teacher, X)
    d_t, d_s = tz.shape[1], extract_features(student, X[:1]).shape[1]
    proj = nn.Linear(d_s, d_t).to(DEVICE)
    opt = torch.optim.Adam(
        list(student.parameters()) + list(proj.parameters()), lr=1e-3
    )
    bce = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS_KD):
        order = np.random.permutation(len(X))
        for st in range(0, len(X), 128):
            idx = order[st : st + 128]
            xb = torch.from_numpy(X[idx]).to(DEVICE)
            yb = torch.from_numpy(y[idx].astype(np.float32)).to(DEVICE)
            with torch.no_grad():
                tf = teacher.forward_features(xb)
            sf = student.forward_features(xb)
            loss_feat = F.mse_loss(proj(sf), tf)
            loss_cls = bce(student(xb), yb)
            loss = loss_cls + 0.5 * loss_feat
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(proj.parameters()), 5.0
            )
            opt.step()
    return student


def transfer_crd(teacher, student, X, y, seed=0):
    # Lightweight contrastive representation distillation.
    set_seed(seed)
    teacher.eval()
    student.train()
    td = extract_features(teacher, X[:1]).shape[1]
    sd = extract_features(student, X[:1]).shape[1]
    proj_t = nn.Linear(td, 64).to(DEVICE)
    proj_s = nn.Linear(sd, 64).to(DEVICE)
    for p in proj_t.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(
        list(student.parameters()) + list(proj_s.parameters()), lr=1e-3
    )
    bce = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS_KD):
        for xb, yb in tensor_batches(X, y, 128, True):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                t = F.normalize(proj_t(teacher.forward_features(xb)), dim=1)
            s = F.normalize(proj_s(student.forward_features(xb)), dim=1)
            logits = (s @ t.T) / 0.2
            labels = torch.arange(len(xb), device=DEVICE)
            loss_crd = F.cross_entropy(logits, labels)
            loss_cls = bce(student(xb), yb)
            loss = loss_cls + 0.5 * loss_crd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return student


def transfer_ensemble_distill(teachers, student, X, y, seed=0):
    set_seed(seed)
    for t in teachers:
        t.eval()
    student.train()
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS_KD):
        for xb, yb in tensor_batches(X, y, 128, True):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                probs = torch.stack([torch.sigmoid(t(xb)) for t in teachers], 0).mean(0)
            slog = student(xb)
            loss = 0.5 * bce(slog, yb) + 0.5 * F.binary_cross_entropy(
                torch.sigmoid(slog), probs
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return student


# =============================== COMBINATION =================================
class CUBase:
    def fit(self, models, X, y):
        self.fitted = True

    def predict(self, models, X):
        raise NotImplementedError


class SimpleAvg(CUBase):
    def predict(self, models, X):
        return np.mean([predict_proba(m, X) for m in models], axis=0)


class ConfidenceWeighted(CUBase):
    """Weight models by calibration-set Brier score, not raw confidence."""

    def fit(self, models, X, y):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        yy = np.asarray(y, dtype=np.float64)[:, None]
        brier = np.mean((ps - yy) ** 2, axis=0)
        # Softmax of negative error; bounded exponentials avoid numerical issues.
        z = -brier / 0.05
        z -= np.max(z)
        self.w = np.exp(z)
        self.w /= self.w.sum()
        self.fitted = True

    def predict(self, models, X):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        w = np.asarray(getattr(self, "w", np.ones(ps.shape[1]) / ps.shape[1]))
        if len(w) != ps.shape[1]:
            raise ValueError("CU/model count changed after calibration")
        return np.sum(ps * w[None, :], axis=1)


class BayesianCombination(CUBase):
    def fit(self, models, X, y):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        eps = 1e-5
        ll = []
        yy = np.asarray(y)
        for j in range(ps.shape[1]):
            p = np.clip(ps[:, j], eps, 1 - eps)
            ll.append(np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
        z = np.exp(np.asarray(ll) - np.max(ll))
        self.w = z / z.sum()
        self.fitted = True

    def predict(self, models, X):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        w = np.asarray(getattr(self, "w", np.ones(ps.shape[1]) / ps.shape[1]))
        if len(w) != ps.shape[1]:
            raise ValueError("CU/model count changed after calibration")
        return np.sum(ps * w[None, :], axis=1)


class StackingCombination(CUBase):
    def fit(self, models, X, y):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        self.meta = LogisticRegression(max_iter=500, random_state=0).fit(ps, y)
        self.fitted = True

    def predict(self, models, X):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        if not getattr(self, "fitted", False):
            return ps.mean(1)
        if ps.shape[1] != self.meta.n_features_in_:
            raise ValueError("CU/model count changed after calibration")
        return self.meta.predict_proba(ps)[:, 1]


class AttentionGating(CUBase):
    def fit(self, models, X, y):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        self.w = np.exp(-np.mean(np.abs(ps - np.asarray(y)[:, None]), axis=0))
        self.w /= self.w.sum()
        self.fitted = True

    def predict(self, models, X):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        w = np.asarray(getattr(self, "w", np.ones(ps.shape[1]) / ps.shape[1]))
        if len(w) != ps.shape[1]:
            raise ValueError("CU/model count changed after calibration")
        return np.sum(ps * w[None, :], axis=1)


class TemperatureScaling(CUBase):
    def fit(self, models, X, y):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        self.temps = []
        yy = torch.from_numpy(np.asarray(y, dtype=np.float32))
        for j in range(ps.shape[1]):
            logit = torch.from_numpy(
                np.log(
                    np.clip(ps[:, j], 1e-5, 1 - 1e-5)
                    / (1 - np.clip(ps[:, j], 1e-5, 1 - 1e-5))
                ).astype(np.float32)
            )
            # Parameterize temperature in log-space so it remains positive and
            # gradients do not die at a hard clamp boundary.
            log_t = nn.Parameter(torch.tensor(0.0))
            opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=30)

            def closure():
                opt.zero_grad(set_to_none=True)
                temperature = torch.exp(
                    torch.clamp(log_t, math.log(0.05), math.log(20.0))
                )
                loss = F.binary_cross_entropy_with_logits(logit / temperature, yy)
                loss.backward()
                return loss

            opt.step(closure)
            self.temps.append(
                float(
                    torch.exp(
                        torch.clamp(log_t.detach(), math.log(0.05), math.log(20.0))
                    )
                )
            )
        self.fitted = True

    def predict(self, models, X):
        ps = np.stack([predict_proba(m, X) for m in models], axis=1)
        ts = np.asarray(getattr(self, "temps", [1.0] * ps.shape[1]))
        if len(ts) != ps.shape[1]:
            raise ValueError("CU/model count changed after calibration")
        logits = np.log(np.clip(ps, 1e-5, 1 - 1e-5) / (1 - np.clip(ps, 1e-5, 1 - 1e-5)))
        return sigmoid_np(np.mean(logits / ts[None, :], axis=1))


class HDT(CUBase):
    def __init__(self, k=5):
        self.k = k

    def predict(self, models, X):
        # Models are ordered [oldest BPU, ..., newest BPU, newest SPU].
        # HDT uses the newest k historical BPUs plus the current SPU.
        use = models[-(self.k + 1) :]
        return np.mean([predict_proba(m, X) for m in use], axis=0)


CU_FACTORY = {
    "HDT": lambda: HDT(k=min(5, MAX_ENSEMBLE - 1)),
    "SimpleAvg": SimpleAvg,
    "ConfWeight": ConfidenceWeighted,
    "Bayesian": BayesianCombination,
    "Stacking": StackingCombination,
    "AttnGate": AttentionGating,
    "TempScale": TemperatureScaling,
}


# ========================== ONLINE SPU =======================================
def online_update(model, X, y, lr=SPU_ONLINE_LR, epochs=1, seed=0):
    """One online mini-batch update with persistent optimizer state.

    The whole revealed batch is updated in one tensor operation.  This avoids
    constructing a DataLoader for every streaming batch and substantially
    reduces Python/CPU overhead while preserving predict-then-update order.
    """
    model.train()
    opt = getattr(model, "_online_opt", None)
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        model._online_opt = opt
    xb = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    yb = torch.as_tensor(y, dtype=torch.float32, device=DEVICE)
    bce = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = bce(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()


# ============================= BASELINES ====================================
def run_batch_baseline(X_train, y_train, X_cal, y_cal, X_test, y_test, seed=0):
    """Static batch baseline: same initial BPU family, TRAIN-only fitting."""
    t0 = time.perf_counter()
    model = safe_train_bpu(
        BPU_MODELS_TO_RUN[0], X_train, y_train, EPOCHS_BPU, seed, bpu=True
    )
    train_seconds = time.perf_counter() - t0
    p_cal = predict_proba(model, X_cal)
    th = tune_threshold(y_cal, p_cal)
    t1 = time.perf_counter()
    p_test = predict_proba(model, X_test)
    test_seconds = time.perf_counter() - t1
    m = binary_metrics(y_test, p_test, th)
    return {
        "acc": m["acc"],
        "f1": m["f1"],
        "roc_auc": m["roc_auc"],
        "pr_auc": m["pr_auc"],
        "threshold": th,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
    }


# ========================== STREAM BASELINES ================================
def run_stream_baseline(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_stream,
    y_stream,
    X_test,
    y_test,
    seed=0,
    warmup=False,
):
    """Test-then-train online baseline using SGDClassifier.

    With warmup=True, CAL labels are revealed before the reported stream and
    used only to warm-start the online learner. Reported STREAM records still
    follow strict predict-then-train ordering.
    """
    from sklearn.linear_model import SGDClassifier

    set_seed(seed)
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed)
    classes = np.array([0, 1], dtype=np.int64)
    clf.partial_fit(X_train[:1], y_train[:1], classes=classes)
    if len(X_train) > 1:
        clf.partial_fit(X_train[1:], y_train[1:])
    if warmup and len(X_cal):
        clf.partial_fit(X_cal, y_cal)

    # Calibrate the decision threshold BEFORE any reported stream labels are
    # revealed. This keeps the stream strictly prequential and prevents
    # post-stream model state from influencing threshold selection.
    th = tune_threshold(y_cal, clf.predict_proba(X_cal)[:, 1])

    stream_pred, stream_y = [], []
    t0 = time.perf_counter()
    for start in range(0, len(y_stream), STREAM_BS):
        end = min(start + STREAM_BS, len(y_stream))
        xb, yb = X_stream[start:end], y_stream[start:end]
        p = clf.predict_proba(xb)[:, 1]
        stream_pred.extend(p.tolist())
        stream_y.extend(yb.tolist())
        clf.partial_fit(xb, yb)
    elapsed = time.perf_counter() - t0
    pt = clf.predict_proba(X_test)[:, 1]
    tm = binary_metrics(y_test, pt, th)
    pm = binary_metrics(np.asarray(stream_y), np.asarray(stream_pred), th)
    return {
        "preq_acc": pm["acc"],
        "preq_f1": pm["f1"],
        "test_acc": tm["acc"],
        "test_f1": tm["f1"],
        "stream_seconds": elapsed,
        "records_per_sec": len(y_stream) / max(elapsed, 1e-9),
        "threshold": th,
        "warmup": warmup,
    }


# ======================= NEURAL STREAM BASELINE =============================
def run_neural_stream_baseline(
    X_train,
    y_train,
    X_cal,
    y_cal,
    X_stream,
    y_stream,
    X_test,
    y_test,
    seed=0,
    drift=False,
):
    """Fixed-complexity neural streaming baseline.

    Uses the fixed MLP student architecture, with exactly the same mini-batch
    size and predict-then-update protocol as StreamWeave, but without a BPU,
    knowledge transfer, ensemble CU, or historical BPU refresh.
    """
    set_seed(seed)
    t0 = time.perf_counter()
    model = safe_train_bpu("MLP", X_train, y_train, EPOCHS_BPU, seed + 500, bpu=False)
    train_seconds = time.perf_counter() - t0
    p_cal = predict_proba(model, X_cal)
    threshold = tune_threshold(y_cal, p_cal)

    preds, labels = [], []
    stream_t0 = time.perf_counter()
    for start in range(0, len(y_stream), STREAM_BS):
        end = min(start + STREAM_BS, len(y_stream))
        xb, yb = X_stream[start:end], y_stream[start:end]
        p = predict_proba(model, xb)
        preds.extend(p.tolist())
        labels.extend(yb.tolist())
        online_update(model, xb, yb, lr=SPU_ONLINE_LR, epochs=1, seed=seed + start)
    stream_seconds = time.perf_counter() - stream_t0

    p_test = predict_proba(model, X_test)
    tm = binary_metrics(y_test, p_test, threshold)
    pm = binary_metrics(np.asarray(labels), np.asarray(preds), threshold)
    return {
        "preq_acc": pm["acc"],
        "preq_f1": pm["f1"],
        "test_acc": tm["acc"],
        "test_f1": tm["f1"],
        "test_roc_auc": tm["roc_auc"],
        "test_pr_auc": tm["pr_auc"],
        "stream_seconds": stream_seconds,
        "records_per_sec": len(y_stream) / max(stream_seconds, 1e-9),
        "train_seconds": train_seconds,
        "threshold": threshold,
        "warmup": False,
        "architecture": "MLP",
        "drift": bool(drift),
    }


# ============================ TRANSFER DRIVER ================================
def fit_transfer(transfer, teacher_models, student, X, y, seed=0):
    if transfer == "None":
        return transfer_none(teacher_models[0], student, X, y, seed)
    if transfer == "StandardKD":
        return transfer_standard_kd(teacher_models[0], student, X, y, seed)
    if transfer == "BornAgain":
        return transfer_born_again(teacher_models[0], student, X, y, seed)
    if transfer == "FitNets":
        return transfer_fitnets(teacher_models[0], student, X, y, seed)
    if transfer == "RKD":
        return transfer_rkd(teacher_models[0], student, X, y, seed)
    if transfer == "CRD":
        return transfer_crd(teacher_models[0], student, X, y, seed)
    if transfer == "EnsembleDistill":
        return transfer_ensemble_distill(teacher_models, student, X, y, seed)
    raise ValueError(transfer)


# ============================ CORE EXPERIMENT ================================
def train_bpu_ensemble(kind, X_train, y_train, seed):
    models = []
    n_models = max(1, min(MAX_ENSEMBLE - 1, 3))
    for i in range(n_models):
        models.append(
            safe_train_bpu(kind, X_train, y_train, EPOCHS_BPU, seed + i * 100, bpu=True)
        )
    return models


def retrain_historical_bpu(kind, X_mem, y_mem, seed):
    return safe_train_bpu(
        kind, X_mem, y_mem, epochs=BPU_REFRESH_EPOCHS, seed=seed, bpu=True
    )


def cu_predict(cu, models, X):
    return np.asarray(cu.predict(models, X), dtype=np.float32)


def evaluate_prequential(
    models,
    cu,
    X_stream,
    y_stream,
    threshold,
    seed=0,
    batch_interval=BATCH_INTERVAL,
    memory_max=MEMORY_MAX,
    bpu_kind=None,
    update_online=True,
    refresh_bpu=True,
):
    """Predict-then-update evaluation with bounded replay memory.

    Every record is predicted before its label is revealed. At each
    batch_interval, a new BPU is trained only on labels already revealed and
    added to the bounded historical ensemble. The CU remains calibrated on CAL
    only, so future stream labels cannot alter its parameters.
    """
    preds, labels = [], []
    memory_x, memory_y = [], []
    refresh_count = 0
    t0 = time.perf_counter()
    for start in range(0, len(y_stream), STREAM_BS):
        end = min(start + STREAM_BS, len(y_stream))
        xb, yb = X_stream[start:end], y_stream[start:end]

        # PREDICT BEFORE REVEALING LABELS.
        p = cu_predict(cu, models, xb)
        preds.extend(p.tolist())
        labels.extend(yb.tolist())

        # Labels are now revealed and may enter replay memory.
        memory_x.append(xb.copy())
        memory_y.append(yb.copy())
        mx = np.concatenate(memory_x)
        my = np.concatenate(memory_y)
        if len(my) > memory_max:
            mx, my = mx[-memory_max:], my[-memory_max:]
        memory_x, memory_y = [mx], [my]

        # Online SPU adaptation.
        if update_online:
            online_update(models[-1], xb, yb, seed=seed + start)

        # Periodic BPU refresh using only labels revealed by this point.
        seen = end
        if (
            refresh_bpu
            and bpu_kind is not None
            and batch_interval > 0
            and seen % batch_interval < len(yb)
        ):
            new_bpu = retrain_historical_bpu(bpu_kind, mx, my, seed + 10000 + seen)
            refresh_count += 1
            # Keep the calibrated ensemble's number and slot ordering stable.
            # Replacing a fixed BPU slot prevents fitted CUs (stacking, weights,
            # temperatures) from receiving a different feature count mid-stream.
            # The final slot is reserved for the continually updated SPU.
            if len(models) < 2:
                raise RuntimeError("Ensemble must contain at least one BPU and one SPU")
            models[0] = new_bpu
            if len(models) > MAX_ENSEMBLE:
                raise RuntimeError("Initial ensemble exceeds MAX_ENSEMBLE")

    elapsed = time.perf_counter() - t0
    p = np.asarray(preds)
    y = np.asarray(labels)
    m = binary_metrics(y, p, threshold)
    m["seconds"] = elapsed
    m["records_per_sec"] = len(y) / max(elapsed, 1e-9)
    m["bpu_refresh_count"] = refresh_count
    return m, p, y


def run_one(
    dataset,
    split,
    bpu_kind,
    spu_kind,
    transfer,
    cu_name,
    seed=0,
    drift=False,
    ablation="full",
):
    """Run one configuration. Ablations: full, no_kd, no_online_spu,
    no_refresh, spu_only. All use the same data split and calibration protocol."""
    valid_ablations = {"full", "no_kd", "no_online_spu", "no_refresh", "spu_only"}
    if ablation not in valid_ablations:
        raise ValueError(f"Unknown ablation: {ablation}")
    # In the full experiment, each seed gets a repeatable stratified data split.
    # The same seed-specific split is reused across all competing configurations.
    if isinstance(split, dict):
        if seed not in split:
            raise KeyError(f"No data split configured for seed={seed}")
        split = split[seed]
    set_seed(seed)
    Xtr, ytr = split.X_train, split.y_train
    Xcal_fit, ycal_fit = split.X_cal_fit, split.y_cal_fit
    Xcal_thresh, ycal_thresh = split.X_cal, split.y_cal
    if Xcal_fit is None or ycal_fit is None:
        raise ValueError(
            "DataSplit lacks a separate calibration-fit partition; rebuild the split with this version."
        )
    Xadapt, yadapt = split.X_adapt.copy(), split.y_adapt.copy()
    Xs, ys = split.X_stream.copy(), split.y_stream.copy()
    Xt, yt = split.X_test.copy(), split.y_test.copy()

    if drift:
        Xadapt, yadapt = inject_drift(
            Xadapt,
            yadapt,
            DRIFT_POS_RATES,
            DRIFT_FEATURE_SHIFT,
            DRIFT_FEATURE_SCALE,
            DRIFT_SEGMENTS,
            seed + 7000,
        )
        Xs, ys = inject_drift(
            Xs,
            ys,
            DRIFT_POS_RATES,
            DRIFT_FEATURE_SHIFT,
            DRIFT_FEATURE_SCALE,
            DRIFT_SEGMENTS,
            seed + 8000,
        )
        # The final test distribution is drifted as well. This makes the drift
        # experiment a genuine out-of-distribution evaluation rather than merely
        # a drifted adaptation stream followed by a clean test set.
        Xt, yt = inject_drift(
            Xt,
            yt,
            DRIFT_POS_RATES,
            DRIFT_FEATURE_SHIFT,
            DRIFT_FEATURE_SCALE,
            DRIFT_SEGMENTS,
            seed + 9000,
        )

    # Initial TRAIN-only construction is timed separately from online processing.
    init_t0 = time.perf_counter()
    bpus = (
        [] if ablation == "spu_only" else train_bpu_ensemble(bpu_kind, Xtr, ytr, seed)
    )
    spu = safe_train_bpu(spu_kind, Xtr, ytr, EPOCHS_BPU, seed + 500, bpu=False)
    effective_transfer = "None" if ablation in {"no_kd", "spu_only"} else transfer
    if ablation != "spu_only":
        spu = fit_transfer(effective_transfer, bpus, spu, Xtr, ytr, seed + 600)
    initial_train_seconds = time.perf_counter() - init_t0
    models = list(bpus) + [spu]

    # CAL is strictly for CU/threshold calibration. SPU-only has no BPU/CU
    # competition, so use the single model directly via SimpleAvg.
    effective_cu_name = "SimpleAvg" if ablation == "spu_only" else cu_name
    cu = CU_FACTORY[effective_cu_name]()
    if effective_cu_name != "HDT":
        cu.fit(models, Xcal_fit, ycal_fit)
    threshold = tune_threshold(ycal_thresh, cu_predict(cu, models, Xcal_thresh))

    # ADAPT is used only for configuration selection. STREAM and ADAPT start
    # from independent copies of exactly the same post-CAL model state.
    adapt_models = copy.deepcopy(models)
    stream_models = copy.deepcopy(models)
    adapt_m, _, _ = evaluate_prequential(
        adapt_models,
        cu,
        Xadapt,
        yadapt,
        threshold,
        seed=seed + 2000,
        batch_interval=BATCH_INTERVAL,
        memory_max=MEMORY_MAX,
        bpu_kind=None if ablation == "spu_only" else bpu_kind,
        update_online=ablation != "no_online_spu",
        refresh_bpu=ablation not in {"no_refresh", "spu_only"},
    )
    preq_m, _, _ = evaluate_prequential(
        stream_models,
        cu,
        Xs,
        ys,
        threshold,
        seed=seed + 3000,
        batch_interval=BATCH_INTERVAL,
        memory_max=MEMORY_MAX,
        bpu_kind=None if ablation == "spu_only" else bpu_kind,
        update_online=ablation != "no_online_spu",
        refresh_bpu=ablation not in {"no_refresh", "spu_only"},
    )

    # TEST is completely locked for model/CU/transfer selection.
    test_t0 = time.perf_counter()
    p_test = cu_predict(cu, stream_models, Xt)
    test_seconds = time.perf_counter() - test_t0
    test_m = binary_metrics(yt, p_test, threshold)

    return {
        "dataset": dataset,
        "seed": seed,
        "bpu": bpu_kind,
        "spu": spu_kind,
        "transfer": effective_transfer,
        "cu": effective_cu_name,
        "drift": bool(drift),
        "ablation": ablation,
        "adapt_acc": adapt_m["acc"],
        "adapt_f1": adapt_m["f1"],
        "adapt_logloss": adapt_m["logloss"],
        "adapt_roc_auc": adapt_m["roc_auc"],
        "adapt_pr_auc": adapt_m["pr_auc"],
        "preq_acc": preq_m["acc"],
        "preq_f1": preq_m["f1"],
        "preq_logloss": preq_m["logloss"],
        "preq_roc_auc": preq_m["roc_auc"],
        "preq_pr_auc": preq_m["pr_auc"],
        "test_acc": test_m["acc"],
        "test_f1": test_m["f1"],
        "test_logloss": test_m["logloss"],
        "test_roc_auc": test_m["roc_auc"],
        "test_pr_auc": test_m["pr_auc"],
        "initial_train_seconds": initial_train_seconds,
        "stream_seconds": preq_m["seconds"],
        "test_seconds": test_seconds,
        "records_per_sec": preq_m["records_per_sec"],
        "bpu_refresh_count": preq_m.get("bpu_refresh_count", 0),
        "threshold": threshold,
        "n_test": len(yt),
        "n_adapt": len(yadapt),
        "n_stream": len(ys),
        "success": True,
    }


# ======================== COMPLETENESS VALIDATION ===========================
def require_complete_phase(rows, phase_name, expected_keys, key_fields):
    """Fail closed unless every planned run identity succeeded exactly once."""
    successful = [r for r in rows if r.get("success") is True]
    actual = []
    for r in successful:
        actual.append(tuple(r.get(k) for k in key_fields))
    expected = [tuple(x) for x in expected_keys]
    from collections import Counter

    actual_counts, expected_counts = Counter(actual), Counter(expected)
    missing = list((expected_counts - actual_counts).elements())
    duplicates = list((actual_counts - expected_counts).elements())
    failed = [r for r in rows if r.get("success") is not True]
    if missing or duplicates or failed:
        detail = {
            "phase": phase_name,
            "expected_runs": len(expected),
            "successful_rows": len(successful),
            "failed_runs": len(failed),
            "missing_run_keys": missing[:10],
            "unexpected_or_duplicate_run_keys": duplicates[:10],
            "errors": [r.get("error", "missing error detail") for r in failed[:5]],
        }
        # Preserve diagnostic rows before aborting so a failed phase is auditable.
        if rows:
            audit_path = os.path.join(SAVE_DIR, f"{phase_name}_incomplete_audit.csv")
            append_csv(audit_path, rows)
        raise RuntimeError(
            f"Refusing model selection from incomplete {phase_name}: {json.dumps(detail, default=str)}"
        )


# ============================ PHASES ========================================
def run_phase1(dataset, split, drift=False):
    rows = []
    for seed in SEEDS:
        for transfer in TRANSFERS_TO_RUN:
            try:
                r = run_one(
                    dataset,
                    split,
                    PHASE1_FIXED_BPU,
                    PHASE1_FIXED_SPU,
                    transfer,
                    "SimpleAvg",
                    seed,
                    drift,
                )
                rows.append(r)
            except Exception as e:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "bpu": PHASE1_FIXED_BPU,
                        "spu": PHASE1_FIXED_SPU,
                        "transfer": transfer,
                        "cu": "SimpleAvg",
                        "drift": bool(drift),
                        "success": False,
                        "error": repr(e),
                    }
                )
                cleanup()
    expected = [
        (dataset, seed, transfer) for seed in SEEDS for transfer in TRANSFERS_TO_RUN
    ]
    require_complete_phase(rows, "phase1", expected, ["dataset", "seed", "transfer"])
    return rows


def select_transfer(rows, dataset):
    df = pd.DataFrame([r for r in rows if r.get("success")])
    df = df[df.dataset == dataset]
    if df.empty:
        errors = [r.get("error") for r in rows if r.get("error")]
        raise RuntimeError(
            f"No successful Phase 1 runs for {dataset}. Errors: {errors[:3]}"
        )
    # IMPORTANT: selection uses CALIBRATION-derived configuration and prequential
    # metrics only. Test F1 is intentionally excluded from selection.
    score = df.groupby("transfer")["adapt_f1"].mean().sort_values(ascending=False)
    return str(score.index[0])


def run_phase2(dataset, split, best_transfer, drift=False):
    rows = []
    for seed in SEEDS:
        for cu in CUS_TO_RUN:
            try:
                r = run_one(
                    dataset,
                    split,
                    PHASE2_FIXED_BPU,
                    PHASE2_FIXED_SPU,
                    best_transfer,
                    cu,
                    seed,
                    drift,
                )
                rows.append(r)
            except Exception as e:
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "bpu": PHASE2_FIXED_BPU,
                        "spu": PHASE2_FIXED_SPU,
                        "transfer": best_transfer,
                        "cu": cu,
                        "drift": bool(drift),
                        "success": False,
                        "error": repr(e),
                    }
                )
                cleanup()
    expected = [(dataset, seed, cu) for seed in SEEDS for cu in CUS_TO_RUN]
    require_complete_phase(rows, "phase2", expected, ["dataset", "seed", "cu"])
    return rows


def select_cu(rows, dataset):
    df = pd.DataFrame([r for r in rows if r.get("success")])
    df = df[df.dataset == dataset]
    if df.empty:
        errors = [r.get("error") for r in rows if r.get("error")]
        raise RuntimeError(
            f"No successful Phase 2 runs for {dataset}. Errors: {errors[:3]}"
        )
    score = df.groupby("cu")["adapt_f1"].mean().sort_values(ascending=False)
    return str(score.index[0])


def run_phase3(dataset, split, best_transfer, best_cu, drift=False):
    rows = []
    for seed in SEEDS:
        for bpu in BPU_MODELS_TO_RUN:
            for spu in SPU_MODELS_TO_RUN:
                try:
                    r = run_one(
                        dataset, split, bpu, spu, best_transfer, best_cu, seed, drift
                    )
                    rows.append(r)
                except Exception as e:
                    rows.append(
                        {
                            "dataset": dataset,
                            "seed": seed,
                            "bpu": bpu,
                            "spu": spu,
                            "transfer": best_transfer,
                            "cu": best_cu,
                            "drift": bool(drift),
                            "success": False,
                            "error": repr(e),
                        }
                    )
                    cleanup()
    expected = [
        (dataset, seed, bpu, spu)
        for seed in SEEDS
        for bpu in BPU_MODELS_TO_RUN
        for spu in SPU_MODELS_TO_RUN
    ]
    require_complete_phase(rows, "phase3", expected, ["dataset", "seed", "bpu", "spu"])
    return rows


def select_phase3(rows, dataset):
    """Select final architecture using ADAPT only; TEST is never inspected."""
    df = pd.DataFrame([r for r in rows if r.get("success")])
    df = df[df.dataset == dataset]
    if df.empty:
        errors = [r.get("error") for r in rows if r.get("error")]
        raise RuntimeError(
            f"No successful Phase 3 runs for {dataset}. Errors: {errors[:3]}"
        )
    # Aggregate across seeds, then choose the configuration with highest mean
    # adaptation F1. This configuration is subsequently evaluated on TEST.
    cols = ["bpu", "spu", "transfer", "cu"]
    score = (
        df.groupby(cols, as_index=False)["adapt_f1"]
        .mean()
        .sort_values("adapt_f1", ascending=False)
    )
    best = score.iloc[0]
    return {c: str(best[c]) for c in cols}


def paired_statistics(selected_rows, batch_rows, stream_rows, neural_rows=None):
    """Paired seed-wise comparisons for the final selected configuration."""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return []
    out = []
    sw = pd.DataFrame(selected_rows)
    ba = pd.DataFrame(batch_rows)
    st = pd.DataFrame(stream_rows)
    ns = pd.DataFrame(neural_rows or [])
    if sw.empty:
        return out
    comparisons = [("Batch", ba, "test_f1"), ("Stream", st, "test_f1")]
    if not ns.empty:
        comparisons.append(("NeuralStream", ns, "test_f1"))
    for ds in sw.dataset.unique():
        a = sw[sw.dataset == ds].sort_values("seed")
        for name, bdf, col in comparisons:
            b = bdf[bdf.dataset == ds].sort_values("seed")
            b = b.sort_values("seed")
            merged = a[["seed", "test_f1"]].merge(
                b[["seed", col]], on="seed", how="inner"
            )
            if len(merged) < 2:
                continue
            x = merged["test_f1_x"].to_numpy()
            y = merged[f"{col}_y"].to_numpy()
            diff = x - y
            try:
                stat, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p = np.nan, 1.0
            out.append(
                {
                    "dataset": ds,
                    "comparison": f"StreamWeave_vs_{name}",
                    "n_pairs": len(diff),
                    "mean_delta_f1": float(np.mean(diff)),
                    "std_delta_f1": float(np.std(diff, ddof=1))
                    if len(diff) > 1
                    else np.nan,
                    "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                    "wilcoxon_p": float(p),
                }
            )
    return out


def summarize(rows):
    df = pd.DataFrame([r for r in rows if r.get("success")])
    if df.empty:
        return pd.DataFrame()
    keys = [
        c
        for c in ["dataset", "bpu", "spu", "transfer", "cu", "drift"]
        if c in df.columns
    ]
    agg = (
        df.groupby(keys)
        .agg(
            n_seeds=("seed", "count"),
            adapt_f1_mean=("adapt_f1", "mean"),
            adapt_f1_std=("adapt_f1", "std"),
            preq_f1_mean=("preq_f1", "mean"),
            preq_f1_std=("preq_f1", "std"),
            preq_acc_mean=("preq_acc", "mean"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            test_acc_mean=("test_acc", "mean"),
            test_roc_auc_mean=("test_roc_auc", "mean"),
            test_pr_auc_mean=("test_pr_auc", "mean"),
            initial_train_seconds_mean=("initial_train_seconds", "mean"),
            seconds_mean=("stream_seconds", "mean"),
            records_per_sec_mean=("records_per_sec", "mean"),
            bpu_refresh_count_mean=("bpu_refresh_count", "mean"),
        )
        .reset_index()
    )
    return agg


# ============================ PLOTTING ======================================
def make_plots(df, out_dir=SAVE_DIR):
    import matplotlib.pyplot as plt

    if df.empty:
        return
    for metric in ["preq_f1_mean", "test_f1_mean"]:
        if metric not in df.columns:
            continue
        for ds in df.dataset.unique():
            sub = df[df.dataset == ds]
            if sub.empty:
                continue
            plt.figure(figsize=(12, 5))
            labels = [f"{r.bpu}/{r.spu}\n{r.transfer}/{r.cu}" for r in sub.itertuples()]
            plt.bar(np.arange(len(sub)), sub[metric].to_numpy())
            plt.xticks(np.arange(len(sub)), labels, rotation=70, ha="right")
            plt.ylabel(metric)
            plt.title(f"StreamWeave — {ds}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{ds}_{metric}.png"), dpi=150)
            plt.close()


# ============================== MAIN ========================================
def run_dataset(dataset, drift=False):
    print(f"\n===== {dataset} | drift={drift} =====")
    raw_X, raw_y = load_raw_dataset(dataset, seed=RANDOM_STATE)
    # Repeated stratified holdout: each seed has its own split, while all model
    # variants for that seed share exactly the same train/cal/adapt/stream/test
    # rows. The raw sample population is fixed for reproducibility.
    splits = {seed: split_and_preprocess(raw_X, raw_y, seed=seed) for seed in SEEDS}
    del raw_X, raw_y
    cleanup()

    # Baselines use the identical TRAIN/CAL/STREAM/TEST partitions. In drift
    # mode, STREAM and TEST are transformed with the same synthetic drift model.
    base_batch, base_stream, base_neural = [], [], []
    for seed in SEEDS:
        split_seed = splits[seed]
        Xtest_b, ytest_b = split_seed.X_test.copy(), split_seed.y_test.copy()
        Xstream_b, ystream_b = split_seed.X_stream.copy(), split_seed.y_stream.copy()
        if drift:
            Xtest_b, ytest_b = inject_drift(
                Xtest_b,
                ytest_b,
                DRIFT_POS_RATES,
                DRIFT_FEATURE_SHIFT,
                DRIFT_FEATURE_SCALE,
                DRIFT_SEGMENTS,
                seed + 9000,
            )
            Xstream_b, ystream_b = inject_drift(
                Xstream_b,
                ystream_b,
                DRIFT_POS_RATES,
                DRIFT_FEATURE_SHIFT,
                DRIFT_FEATURE_SCALE,
                DRIFT_SEGMENTS,
                seed + 8000,
            )
        b = run_batch_baseline(
            split_seed.X_train,
            split_seed.y_train,
            split_seed.X_cal,
            split_seed.y_cal,
            Xtest_b,
            ytest_b,
            seed,
        )
        sb = run_stream_baseline(
            split_seed.X_train,
            split_seed.y_train,
            split_seed.X_cal,
            split_seed.y_cal,
            Xstream_b,
            ystream_b,
            Xtest_b,
            ytest_b,
            seed,
            warmup=False,
        )
        nb = run_neural_stream_baseline(
            split_seed.X_train,
            split_seed.y_train,
            split_seed.X_cal,
            split_seed.y_cal,
            Xstream_b,
            ystream_b,
            Xtest_b,
            ytest_b,
            seed,
            drift=drift,
        )
        base_batch.append(
            {
                "dataset": dataset,
                "seed": seed,
                "baseline": "Batch",
                "test_acc": b["acc"],
                "test_f1": b["f1"],
                "test_roc_auc": b["roc_auc"],
                "test_pr_auc": b["pr_auc"],
                "threshold": b["threshold"],
                "train_seconds": b["train_seconds"],
                "test_seconds": b["test_seconds"],
                "drift": drift,
            }
        )
        base_stream.append(
            {
                "dataset": dataset,
                "seed": seed,
                "baseline": "Stream",
                **sb,
                "drift": drift,
            }
        )
        base_neural.append(
            {
                "dataset": dataset,
                "seed": seed,
                "baseline": "NeuralStream",
                **nb,
                "drift": drift,
            }
        )

    p1 = run_phase1(dataset, splits, drift)
    best_transfer = select_transfer(p1, dataset)
    print("Phase 1 selected transfer:", best_transfer)

    p2 = run_phase2(dataset, splits, best_transfer, drift)
    best_cu = select_cu(p2, dataset)
    print("Phase 2 selected CU:", best_cu)

    p3 = run_phase3(dataset, splits, best_transfer, best_cu, drift)
    selected = select_phase3(p3, dataset)
    print("Phase 3 selected architecture:", selected)

    selected_rows = [
        r
        for r in p3
        if r.get("success")
        and all(r[k] == selected[k] for k in ["bpu", "spu", "transfer", "cu"])
    ]

    # Component ablations use the selected configuration and identical split.
    ablation_rows = []
    for ablation in ["full", "no_kd", "no_online_spu", "no_refresh", "spu_only"]:
        for seed in SEEDS:
            try:
                ar = run_one(
                    dataset,
                    splits,
                    selected["bpu"],
                    selected["spu"],
                    selected["transfer"],
                    selected["cu"],
                    seed,
                    drift,
                    ablation=ablation,
                )
                ablation_rows.append(ar)
            except Exception as e:
                ablation_rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "bpu": selected["bpu"],
                        "spu": selected["spu"],
                        "transfer": selected["transfer"],
                        "cu": selected["cu"],
                        "drift": bool(drift),
                        "ablation": ablation,
                        "success": False,
                        "error": repr(e),
                    }
                )
                cleanup()
    # Ablations intentionally change the effective transfer/CU for some variants;
    # validate their stable identity using the requested ablation and seed instead.
    require_complete_phase(
        ablation_rows,
        "ablations",
        [
            (dataset, seed, a)
            for a in ["full", "no_kd", "no_online_spu", "no_refresh", "spu_only"]
            for seed in SEEDS
        ],
        ["dataset", "seed", "ablation"],
    )
    return {
        "dataset": dataset,
        "best_transfer": best_transfer,
        "best_cu": best_cu,
        "selected": selected,
        "baseline_batch": base_batch,
        "baseline_stream": base_stream,
        "baseline_neural": base_neural,
        "phase1": p1,
        "phase2": p2,
        "phase3": p3,
        "selected_rows": selected_rows,
        "ablations": ablation_rows,
    }


def save_experiment(result, drift=False):
    suffix = "_drift" if drift else ""
    ds = result["dataset"]
    append_csv(os.path.join(SAVE_DIR, f"phase1{suffix}.csv"), result["phase1"])
    append_csv(os.path.join(SAVE_DIR, f"phase2{suffix}.csv"), result["phase2"])
    append_csv(os.path.join(SAVE_DIR, f"phase3{suffix}.csv"), result["phase3"])
    append_csv(
        os.path.join(SAVE_DIR, f"batch_baseline{suffix}.csv"), result["baseline_batch"]
    )
    append_csv(
        os.path.join(SAVE_DIR, f"stream_baseline{suffix}.csv"),
        result["baseline_stream"],
    )
    append_csv(
        os.path.join(SAVE_DIR, f"neural_stream_baseline{suffix}.csv"),
        result["baseline_neural"],
    )
    append_csv(
        os.path.join(SAVE_DIR, f"selected_final{suffix}.csv"), result["selected_rows"]
    )
    append_csv(
        os.path.join(SAVE_DIR, f"ablations{suffix}.csv"), result.get("ablations", [])
    )
    stats = paired_statistics(
        result["selected_rows"],
        result["baseline_batch"],
        result["baseline_stream"],
        result["baseline_neural"],
    )
    append_csv(os.path.join(SAVE_DIR, f"paired_statistics{suffix}.csv"), stats)
    meta = {
        "dataset": ds,
        "best_transfer": result["best_transfer"],
        "best_cu": result["best_cu"],
        "selected": result["selected"],
        "sample_size": SAMPLE_SIZE,
        "seeds": SEEDS,
        "train_frac": TRAIN_FRAC,
        "cal_frac": CAL_FRAC,
        "cal_fit_frac": CAL_FIT_FRAC,
        "cal_thresh_frac": CAL_THRESH_FRAC,
        "adapt_frac": ADAPT_FRAC,
        "stream_eval_frac": STREAM_FRAC,
        "test_frac": TEST_FRAC,
        "stream_batch_size": STREAM_BS,
        "batch_interval": BATCH_INTERVAL,
        "bpu_refresh_epochs": BPU_REFRESH_EPOCHS,
        "memory_max": MEMORY_MAX,
        "neural_stream_baseline": "fixed MLP, same mini-batch online protocol",
        "max_ensemble": MAX_ENSEMBLE,
        "device": DEVICE,
        "drift": drift,
        "drift_pos_rates": DRIFT_POS_RATES,
        "drift_feature_shift": DRIFT_FEATURE_SHIFT,
        "drift_feature_scale": DRIFT_FEATURE_SCALE,
        "drift_segments": DRIFT_SEGMENTS,
        "selection_metric": "mean_adapt_f1",
        "test_used_for_selection": False,
        "split_protocol": "seed-specific repeated stratified holdout; configurations paired within seed",
        "seeds_are_independent_samples_for_inference": False,
        "hdbs_dt_included": False,
        "methodology_note": (
            "Neural reimplementation candidate of HDBS/GHDBS; KD replaces the "
            "tree-model converter. HDT here is a probability-averaging analogue, "
            "not the original decision-tree HDT algorithm. Synthetic drift is "
            "controlled class-prior resampling plus feature perturbation, not natural drift."
        ),
        "python_version": __import__("sys").version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
    }
    with open(os.path.join(SAVE_DIR, f"{ds}_config{suffix}.json"), "w") as f:
        json.dump(meta, f, indent=2)


def run_nsl_official(selected_result):
    """External NSL-KDD evaluation: KDDTrain+ split 90/5/5, test on KDDTest+.

    The 90% training subset fits preprocessing and neural models. CAL_FIT fits
    CU parameters; CAL_THRESH selects the F1 threshold. KDDTest+ is never used
    for fitting, calibration, threshold selection, or model selection.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from pathlib import Path
    import hashlib

    out_path = os.path.join(SAVE_DIR, "nsl_official.csv")
    train_url = (
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
    )
    test_url = (
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
    )
    columns = [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "land",
        "wrong_fragment",
        "urgent",
        "hot",
        "num_failed_logins",
        "logged_in",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_file_creations",
        "num_shells",
        "num_access_files",
        "num_outbound_cmds",
        "is_host_login",
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "dst_host_serror_rate",
        "dst_host_srv_serror_rate",
        "dst_host_rerror_rate",
        "dst_host_srv_rerror_rate",
        "label",
        "difficulty",
    ]
    cache = Path(DATA_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    train_path, test_path = cache / "KDDTrain+.txt", cache / "KDDTest+.txt"
    if not train_path.exists():
        pd.read_csv(train_url, header=None, names=columns).to_csv(
            train_path, index=False, header=False
        )
    if not test_path.exists():
        pd.read_csv(test_url, header=None, names=columns).to_csv(
            test_path, index=False, header=False
        )
    train_df = pd.read_csv(train_path, header=None, names=columns)
    test_df = pd.read_csv(test_path, header=None, names=columns)
    y_all = (
        (train_df["label"].astype(str).str.lower() != "normal")
        .astype(np.int64)
        .to_numpy()
    )
    y_test = (
        (test_df["label"].astype(str).str.lower() != "normal")
        .astype(np.int64)
        .to_numpy()
    )
    X_all = train_df.drop(columns=["label", "difficulty"])
    X_test_raw = test_df.drop(columns=["label", "difficulty"])
    cat_cols = ["protocol_type", "service", "flag"]
    numeric_cols = [c for c in X_all.columns if c not in cat_cols]
    rows = []
    selected = selected_result["selected"]
    transfer = selected_result["best_transfer"]
    cu_name = selected_result["best_cu"]

    for seed in SEEDS:
        # Deterministic, stratified 90/5/5 partition of KDDTrain+.
        idx = np.arange(len(y_all))
        idx_train, idx_cal = train_test_split(
            idx, test_size=0.10, stratify=y_all, random_state=seed
        )
        idx_cal_fit, idx_cal_thresh = train_test_split(
            idx_cal, test_size=0.50, stratify=y_all[idx_cal], random_state=seed + 101
        )
        Xtr_df = X_all.iloc[idx_train].copy()
        Xcf_df = X_all.iloc[idx_cal_fit].copy()
        Xct_df = X_all.iloc[idx_cal_thresh].copy()

        # Numeric imputation and categorical vocabulary are fit on TRAIN only.
        medians = Xtr_df[numeric_cols].apply(pd.to_numeric, errors="coerce").median()

        def prep_numeric(frame):
            return (
                frame[numeric_cols]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(medians)
            )

        def prep_cats(frame):
            return frame[cat_cols].fillna("__MISSING__").astype(str)

        Xtr_num = prep_numeric(Xtr_df)
        Xcf_num = prep_numeric(Xcf_df)
        Xct_num = prep_numeric(Xct_df)
        Xte_num = prep_numeric(X_test_raw)
        Xtr_cat = pd.get_dummies(prep_cats(Xtr_df), columns=cat_cols, dtype=np.float32)
        schema = list(Xtr_cat.columns)

        def cats_transform(frame):
            enc = pd.get_dummies(prep_cats(frame), columns=cat_cols, dtype=np.float32)
            return enc.reindex(columns=schema, fill_value=0).to_numpy(np.float32)

        A_tr = np.hstack([Xtr_num.to_numpy(np.float32), Xtr_cat.to_numpy(np.float32)])
        A_cf = np.hstack([Xcf_num.to_numpy(np.float32), cats_transform(Xcf_df)])
        A_ct = np.hstack([Xct_num.to_numpy(np.float32), cats_transform(Xct_df)])
        A_te = np.hstack([Xte_num.to_numpy(np.float32), cats_transform(X_test_raw)])
        scaler = StandardScaler().fit(A_tr)
        Xtr = scaler.transform(A_tr).astype(np.float32)
        Xcf = scaler.transform(A_cf).astype(np.float32)
        Xct = scaler.transform(A_ct).astype(np.float32)
        Xte = scaler.transform(A_te).astype(np.float32)
        ytr, ycf, yct = y_all[idx_train], y_all[idx_cal_fit], y_all[idx_cal_thresh]

        set_seed(seed)
        bpus = train_bpu_ensemble(selected["bpu"], Xtr, ytr, seed)
        spu = safe_train_bpu(
            selected["spu"], Xtr, ytr, EPOCHS_BPU, seed + 500, bpu=False
        )
        spu = fit_transfer(transfer, bpus, spu, Xtr, ytr, seed + 600)
        models = list(bpus) + [spu]
        cu = CU_FACTORY[cu_name]()
        if cu_name != "HDT":
            cu.fit(models, Xcf, ycf)
        p_thresh = cu_predict(cu, models, Xct)
        threshold = tune_threshold(yct, p_thresh)
        p_test = cu_predict(cu, models, Xte)
        metrics = binary_metrics(y_test, p_test, threshold)
        rows.append(
            {
                "dataset": "NSL-KDD",
                "protocol": "official_KDDTrainPlus_to_KDDTestPlus",
                "seed": seed,
                "train_samples": int(len(idx_train)),
                "cal_fit_samples": int(len(idx_cal_fit)),
                "cal_thresh_samples": int(len(idx_cal_thresh)),
                "test_samples": int(len(y_test)),
                "bpu": selected["bpu"],
                "spu": selected["spu"],
                "transfer": transfer,
                "cu": cu_name,
                "threshold": float(threshold),
                "test_acc": metrics["acc"],
                "test_f1": metrics["f1"],
                "test_logloss": metrics["logloss"],
                "test_roc_auc": metrics["roc_auc"],
                "test_pr_auc": metrics["pr_auc"],
                "success": True,
                "train_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
                "test_sha256": hashlib.sha256(test_path.read_bytes()).hexdigest(),
            }
        )
        print(
            f"[NSL-OFFICIAL] seed={seed} F1={metrics['f1']:.4f} ROC-AUC={metrics['roc_auc']:.4f}"
        )
    append_csv(out_path, rows)
    return rows


def main():
    all_results = []
    drift_modes = [False, True] if ENABLE_DRIFT else [False]
    for drift in drift_modes:
        for ds in DATASETS_TO_RUN:
            try:
                result = run_dataset(ds, drift=drift)
                save_experiment(result, drift=drift)
                all_results.append(result)
                print(f"Saved {ds}")
            except Exception as e:
                print(f"FAILED {ds}: {e}")
                import traceback

                traceback.print_exc()
            finally:
                cleanup()

    for result in all_results:
        print("\n", result["dataset"])
        print("  transfer:", result["best_transfer"])
        print("  CU:", result["best_cu"])
        print("  selected:", result["selected"])
        s = summarize(result["phase3"])
        if not s.empty:
            sel = s[
                (s.bpu == result["selected"]["bpu"])
                & (s.spu == result["selected"]["spu"])
                & (s.transfer == result["selected"]["transfer"])
                & (s.cu == result["selected"]["cu"])
            ]
            if not sel.empty:
                print(
                    sel[
                        [
                            "bpu",
                            "spu",
                            "transfer",
                            "cu",
                            "adapt_f1_mean",
                            "preq_f1_mean",
                            "test_f1_mean",
                            "test_acc_mean",
                            "seconds_mean",
                            "records_per_sec_mean",
                        ]
                    ].to_string(index=False)
                )
            make_plots(s)

    clean_nsl = next(
        (
            r
            for r in all_results
            if r.get("dataset") == "NSL-KDD" and not r.get("drift", False)
        ),
        None,
    )
    official_nsl_rows = []
    if clean_nsl is not None:
        try:
            official_nsl_rows = run_nsl_official(clean_nsl)
        except Exception as e:
            print(f"OFFICIAL NSL-KDD EVALUATION FAILED: {e}")
            with open(os.path.join(SAVE_DIR, "nsl_official_error.txt"), "w") as f:
                f.write(repr(e))

    manifest = {
        "protocol": "research_candidate_v15_calibration_split_completeness_gated",
        "sample_size": SAMPLE_SIZE,
        "seeds": SEEDS,
        "datasets": DATASETS_TO_RUN,
        "drift_modes": drift_modes,
        "stream_batch_size": STREAM_BS,
        "batch_interval": BATCH_INTERVAL,
        "bpu_refresh_epochs": BPU_REFRESH_EPOCHS,
        "selection_metric": "mean_adapt_f1",
        "test_used_for_selection": False,
        "hdbs_dt_included": False,
        "note": "Research candidate, not a guarantee of publication readiness. End-to-end stream timing includes online SPU updates and periodic BPU refreshes; stream length is 20% of the sampled records.",
        "python_version": __import__("sys").version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": __import__("sklearn").__version__,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "ablations": ["full", "no_kd", "no_online_spu", "no_refresh", "spu_only"],
        "official_nsl_rows_written": len(official_nsl_rows),
        "official_nsl_protocol": "KDDTrain+ stratified 90/5/5; preprocessing/model fit on 90%; CU fit on CAL_FIT; threshold on CAL_THRESH; final evaluation on KDDTest+.",
    }
    with open(os.path.join(SAVE_DIR, "FINAL_MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\nResults directory:", SAVE_DIR)


if __name__ == "__main__":
    main()
