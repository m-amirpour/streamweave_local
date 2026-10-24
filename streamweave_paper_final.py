"""StreamWeave: resource-bounded neural BPU/SPU/KD research runner.

Neural research analogue of Pishgoo et al. (Information Sciences, 2021),
"A hybrid distributed batch-stream processing approach for anomaly detection".

Protocol safeguards:
- The source dataset is split ONCE before screening. Test data are disjoint from
  screening, configuration selection, calibration, and prequential stream data.
- Preprocessing is fitted on the training partition only.
- A budgeted, deterministic pairwise-coverage pilot explores BPU x SPU x
  transfer combinations on each dataset; every CU is evaluated for each selected
  neural combination. The screen is followed by multi-seed confirmation and a
  single reserved final-test evaluation of the selected configuration.
- The pilot uses smaller data/epoch budgets, and the screened transfer="None"
  control receives the same number of additional optimizer epochs as KD methods.
- Stream prediction precedes label reveal/update. Natural chronological data keep
  their row order; no synthetic drift is injected.
- The neural BPU-to-SPU converter is knowledge distillation at BPU refresh time.
- CSV outputs are resumable and carry source/config signatures.

This is a reproducible experimental implementation, not a claim of paper-ready
results by itself. Validate each dataset end-to-end, inspect failure logs and
partition profiles, pre-register/justify the analysis, and report the emitted
protocol manifest and all failed runs before making paper claims.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import torch.nn.functional as F
from scipy.io import arff as scipy_arff
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------- Configuration ------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    Path(os.environ.get("STREAMWEAVE_DATA_DIR", BASE_DIR / "data"))
    .expanduser()
    .resolve()
)
SAVE_DIR = (
    Path(
        os.environ.get(
            "STREAMWEAVE_RESULTS_DIR", BASE_DIR / "results" / "streamweave_paper"
        )
    )
    .expanduser()
    .resolve()
)

_default_datasets = (
    "Covertype,Skin,N-BaIoT,NSL-KDD,CreditCardFraud,Adult,BankMarketing,Electricity"
)
DATASETS_TO_RUN = [
    x.strip()
    for x in os.environ.get("STREAMWEAVE_DATASETS", _default_datasets).split(",")
    if x.strip()
]
BPU_MODELS = ["MLP", "DeepAE", "ResidualMLP", "WideMLP", "FeatureAttention"]
SPU_MODELS = ["SmallMLP", "MLP", "ResidualMLP", "FeatureAttention", "GRU", "TCN"]
TRANSFERS = [
    "None",
    "StandardKD",
    "RKD",
    "BornAgainStyleKD",
    "FitNets",
    "ContrastiveRepKD",
    "EnsembleDistill",
]
CUS = [
    "HDT_Analog",
    "SimpleAvg",
    "ConfWeight",
    "Bayesian",
    "Stacking",
    "ErrorWeighted",
    "TempScale",
]

# Laptop-friendly, staged search defaults. Override with STREAMWEAVE_* variables
# after measuring throughput on the target machine. Screening never touches TEST.
SCREEN_TRAIN_SIZE = int(os.environ.get("STREAMWEAVE_SCREEN_TRAIN_SIZE", "2000"))
CONFIRM_TRAIN_SIZE = int(os.environ.get("STREAMWEAVE_CONFIRM_TRAIN_SIZE", "8000"))
SCREEN_EPOCHS = int(os.environ.get("STREAMWEAVE_SCREEN_EPOCHS", "1"))
SCREEN_KD_EPOCHS = int(os.environ.get("STREAMWEAVE_SCREEN_KD_EPOCHS", "1"))
CONFIRM_EPOCHS = int(os.environ.get("STREAMWEAVE_CONFIRM_EPOCHS", "5"))
CONFIRM_KD_EPOCHS = int(os.environ.get("STREAMWEAVE_CONFIRM_KD_EPOCHS", "1"))
SCREEN_NEURAL_BUDGET = int(os.environ.get("STREAMWEAVE_SCREEN_NEURAL_BUDGET", "43"))
SCREEN_MAX_FAILURE_FRACTION = float(
    os.environ.get("STREAMWEAVE_SCREEN_MAX_FAILURE_FRACTION", "0.01")
)
if not 0.0 <= SCREEN_MAX_FAILURE_FRACTION <= 1.0:
    raise ValueError("STREAMWEAVE_SCREEN_MAX_FAILURE_FRACTION must be in [0, 1]")
REFRESH_EPOCHS = int(os.environ.get("STREAMWEAVE_REFRESH_EPOCHS", "2"))
ONLINE_EPOCHS = int(os.environ.get("STREAMWEAVE_ONLINE_EPOCHS", "1"))
SCREEN_EVAL_MAX = int(os.environ.get("STREAMWEAVE_SCREEN_EVAL_MAX", "2500"))
SCREEN_TOP_K = int(os.environ.get("STREAMWEAVE_TOP_K", "3"))
SEEDS = [
    int(x) for x in os.environ.get("STREAMWEAVE_SEEDS", "0,1,2").split(",") if x.strip()
]
if not SEEDS or len(SEEDS) != len(set(SEEDS)):
    raise ValueError("STREAMWEAVE_SEEDS must contain at least one unique integer seed")
if min(SCREEN_TRAIN_SIZE, CONFIRM_TRAIN_SIZE, SCREEN_EPOCHS, CONFIRM_EPOCHS) < 1:
    raise ValueError("Train sizes and epoch budgets must be positive integers")
if SCREEN_KD_EPOCHS < 0 or CONFIRM_KD_EPOCHS < 0:
    raise ValueError("KD epoch budgets must be non-negative")
BPU_HISTORY_SIZE = 5  # k historical BPU predictors + the active SPU
STREAM_BATCH_SIZE = int(os.environ.get("STREAMWEAVE_STREAM_BATCH_SIZE", "256"))
BATCH_INTERVAL = int(os.environ.get("STREAMWEAVE_BATCH_INTERVAL", "8192"))
MEMORY_MAX = int(os.environ.get("STREAMWEAVE_MEMORY_MAX", "4096"))
ONLINE_LR = float(os.environ.get("STREAMWEAVE_ONLINE_LR", "0.001"))
POS_WEIGHT_CAP = float(os.environ.get("STREAMWEAVE_POS_WEIGHT_CAP", "100.0"))
BASE_LR = float(os.environ.get("STREAMWEAVE_LR", "0.001"))
WEIGHT_DECAY = float(os.environ.get("STREAMWEAVE_WEIGHT_DECAY", "0.0001"))
NUM_WORKERS = int(os.environ.get("STREAMWEAVE_NUM_WORKERS", "0"))
MASTER_SEED = int(os.environ.get("STREAMWEAVE_MASTER_SEED", "20261010"))
SCREEN_SEED = int(os.environ.get("STREAMWEAVE_SCREEN_SEED", str(MASTER_SEED)))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fractions of the entire source dataset. For temporal datasets these are
# contiguous chronological segments; for non-temporal datasets they are
# disjoint stratified partitions.
FRACTIONS = {
    "train": 0.60,
    "cal_fit": 0.05,
    "cal_thresh": 0.05,
    "adapt": 0.10,
    "stream": 0.10,
    "test": 0.10,
}


def _screen_neural_combinations() -> list[tuple[str, str, str]]:
    """Return a deterministic low-budget pairwise covering design.

    The 5 x 6 x 7 neural grid has 210 possible triples. A greedy covering-array
    pass covers every BPU-SPU, BPU-transfer, and SPU-transfer level pair first;
    this requires 43 triples for the current factors. The remaining budget (48
    by default) is filled with a seeded sample of the uncovered triples. All CU
    methods are then evaluated for each chosen triple, since CU fitting is cheap
    relative to training a new student. The design is repeated independently per
    dataset using that dataset's development partitions only.
    """
    all_combos = [
        (bpu, spu, transfer)
        for bpu in BPU_MODELS
        for spu in SPU_MODELS
        for transfer in TRANSFERS
    ]

    def pairs(combo: tuple[str, str, str]) -> set[tuple[str, str, str]]:
        bpu, spu, transfer = combo
        return {
            ("bpu_spu", bpu, spu),
            ("bpu_transfer", bpu, transfer),
            ("spu_transfer", spu, transfer),
        }

    uncovered = {p for c in all_combos for p in pairs(c)}
    selected: list[tuple[str, str, str]] = []
    selected_set: set[tuple[str, str, str]] = set()
    while uncovered:
        best_combo = None
        best_gain = -1
        for combo in all_combos:
            if combo in selected_set:
                continue
            gain = len(pairs(combo) & uncovered)
            if gain > best_gain:
                best_combo, best_gain = combo, gain
        if best_combo is None or best_gain <= 0:
            raise RuntimeError("Could not construct pairwise screening design")
        selected.append(best_combo)
        selected_set.add(best_combo)
        uncovered.difference_update(pairs(best_combo))

    requested = max(1, min(int(SCREEN_NEURAL_BUDGET), len(all_combos)))
    effective_budget = max(requested, len(selected))  # never sacrifice pair coverage
    remaining = [c for c in all_combos if c not in selected_set]
    random.Random(SCREEN_SEED).shuffle(remaining)
    selected.extend(remaining[: effective_budget - len(selected)])
    return selected


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    probe = SAVE_DIR / ".write_test"
    probe.write_text("ok\n", encoding="utf-8")
    probe.unlink()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001, S110 — cudnn flags are best-effort on CPU builds
        pass


def cleanup() -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_input_file(path: Path, source_url: str | None = None) -> str:
    """Record a source checksum, reusing it when path, mtime, and size match.

    The metadata cache avoids re-reading multi-GB files on every invocation.
    This assumes filesystems update mtime when contents change; delete the
    manifest entry or manifest to force a full checksum verification.
    """
    path = path.resolve()
    stat = path.stat()
    manifest_path = DATA_DIR / "source_sha256_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(
                f"Cannot read source SHA-256 manifest {manifest_path}: {exc}"
            ) from exc
    else:
        manifest = {"format_version": 1, "files": {}}
    files = manifest.setdefault("files", {})
    key = str(path)
    prior = files.get(key)
    prior = prior if isinstance(prior, dict) else {}
    metadata_matches = (
        prior.get("size_bytes") == stat.st_size
        and prior.get("mtime_ns") == stat.st_mtime_ns
        and isinstance(prior.get("sha256"), str)
    )
    digest = prior["sha256"] if metadata_matches else _sha256_file(path)
    expected = prior.get("sha256")
    if expected and str(expected).lower() != digest.lower():
        raise RuntimeError(
            f"Source file checksum mismatch for {path}: expected {expected}, actual {digest}. "
            f"If the data change is intentional, delete this path entry from {manifest_path} "
            f"to re-pin the file on the next run (or remove the manifest to rehash all inputs)."
        )
    files[key] = {
        "sha256": digest,
        "source_url": source_url or prior.get("source_url"),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    tmp = manifest_path.with_name(manifest_path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return digest


# -------------------------------- Data loading ------------------------------
def _norm_label(s: Any) -> str:
    return str(s).strip().lower().rstrip(".")


def _first_csv_row_is_data(path: Path, min_numeric_fraction: float = 0.80) -> bool:
    """Detect a headerless numeric CSV by inspecting its first row as data.

    This is intentionally schema-agnostic: standard N-BaIoT files often have
    named feature headers, while some repackaged files omit them.
    """
    try:
        first = pd.read_csv(path, header=None, nrows=1, low_memory=False)
    except Exception:
        return False
    if first.empty:
        return False
    row = first.iloc[0]
    numeric = pd.to_numeric(row, errors="coerce")

    return bool(
        len(row)
        and float(np.count_nonzero(~pd.isna(numeric)) / len(row))
        >= min_numeric_fraction
    )



def _binary_labels(y: Iterable[Any], positive: str | None = None) -> np.ndarray:
    s = pd.Series(y).astype(str).map(_norm_label)
    vals = sorted(s.unique().tolist())
    if positive is not None:
        p = _norm_label(positive)
        aliases = {p}
        try:
            numeric_positive = float(p)
            if numeric_positive.is_integer():
                aliases.add(f"{int(numeric_positive)}.0")
        except ValueError:
            pass
        if p == "1":
            aliases.update({"1.0", "true", "yes", "fraud"})
        if p in {">50k", ">50k."}:
            aliases.update({">50k", "1", "true"})
        if p == "yes":
            aliases.update({"yes", "y", "1", "true"})
        if p == "up":
            aliases.update({"up", "1", "true"})
        out = s.isin(aliases).astype(np.int64).to_numpy()
        if out.sum() == 0:
            raise ValueError(
                f"Positive class {positive!r} not found; observed labels={vals[:20]}"
            )
        return out
    if len(vals) != 2:
        raise ValueError(
            f"Expected a binary target, got {len(vals)} classes: {vals[:20]}"
        )
    return (s.to_numpy() == vals[-1]).astype(np.int64)


def _read_skin() -> tuple[pd.DataFrame, np.ndarray]:
    paths = [
        DATA_DIR / "Skin_NonSkin.txt",
        DATA_DIR / "skin.csv",
        BASE_DIR / "skin.csv",
        Path("Skin_NonSkin.txt"),
    ]
    df = None
    for path in paths:
        if not path.exists():
            continue
        _record_input_file(path)
        if path.suffix.lower() == ".txt":
            df = pd.read_csv(path, sep=r"\s+|\t|,", engine="python", header=None)
        elif _first_csv_row_is_data(path, min_numeric_fraction=0.90):
            # Headerless skin.csv: preserve the first observation.
            df = pd.read_csv(path, header=None, sep=None, engine="python")
        else:
            df = pd.read_csv(path, sep=None, engine="python")
        break
    if df is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt"
        target = DATA_DIR / "Skin_NonSkin.txt"
        temp = target.with_suffix(".download")
        try:
            with (
                urllib.request.urlopen(url, timeout=45) as resp,
                open(temp, "wb") as fout,
            ):
                fout.write(resp.read())
            if temp.stat().st_size < 1000:
                raise RuntimeError("Downloaded Skin dataset is unexpectedly small")
            os.replace(temp, target)
            _record_input_file(target, source_url=url)
            df = pd.read_csv(target, sep=r"\s+|\t|,", engine="python", header=None)
        except Exception as exc:
            temp.unlink(missing_ok=True)
            raise RuntimeError(
                f"Skin data unavailable; place Skin_NonSkin.txt in {DATA_DIR}: {exc}"
            ) from exc
    if df.shape[1] < 2:
        raise ValueError("Skin data must contain at least one feature and one label")
    # UCI Skin Segmentation uses labels 1 and 2. Label 2 (non-skin) is treated
    # as the positive/anomalous class for consistency with the existing protocol.
    y = _binary_labels(df.iloc[:, -1].to_numpy(), positive="2")
    X = df.iloc[:, :-1].copy()
    X.columns = [f"feature_{i}" for i in range(X.shape[1])]
    return X.reset_index(drop=True), y


def _read_nsl_kdd() -> tuple[pd.DataFrame, np.ndarray]:
    names = [
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
    candidates = [
        DATA_DIR / "KDDTrain+.txt",
        DATA_DIR / "KDDTrain+.csv",
        DATA_DIR / "KDDTrain+",
        BASE_DIR / "KDDTrain+.txt",
    ]
    data_path = next((p for p in candidates if p.exists()), None)
    source_url = None
    if data_path is None:
        source_url = os.environ.get(
            "STREAMWEAVE_NSLKDD_URL",
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt",
        )
        data_path = DATA_DIR / "KDDTrain+.txt"
        tmp_path = data_path.with_name(data_path.name + f".download.{os.getpid()}")
        try:
            with (
                urllib.request.urlopen(source_url, timeout=60) as response,
                tmp_path.open("wb") as out,
            ):
                shutil.copyfileobj(response, out)
            if tmp_path.stat().st_size < 1_000_000:
                raise RuntimeError(
                    f"Downloaded NSL-KDD file unexpectedly small: {tmp_path.stat().st_size} bytes"
                )
            os.replace(tmp_path, data_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download NSL-KDD source from {source_url}: {exc}"
            ) from exc
    _record_input_file(data_path, source_url=source_url)
    df = pd.read_csv(data_path, header=None, names=names)
    if df.shape[1] != len(names):
        raise ValueError(
            f"NSL-KDD schema mismatch: expected {len(names)} columns, got {df.shape[1]}"
        )
    y = (
        (df["label"].astype(str).str.lower().str.rstrip(".") != "normal")
        .astype(np.int64)
        .to_numpy()
    )
    X = df.drop(columns=["label", "difficulty"])
    return X.reset_index(drop=True), y

def _nbaiot_explicit_labels(series: pd.Series, path: Path) -> np.ndarray:
    """Convert a supplied label column into benign=0 / attack=1, or fail loudly."""
    raw = series.copy()
    numeric_values = pd.to_numeric(raw, errors="coerce")
    numeric_array = np.asarray(numeric_values)

    if len(raw) and bool(np.isfinite(
        np.asarray(numeric_array, dtype=np.float64)
    ).all()):
        values = set(float(v) for v in np.unique(numeric_array))

        if values.issubset({0.0, 1.0}):
            out: np.ndarray = (numeric_array > 0).astype(np.int64)
        elif 0.0 in values and len(values) >= 2:
            # Multi-class labels with zero as benign and positive IDs as attacks.
            out = (numeric_array > 0).astype(np.int64)
        else:
            raise ValueError(
                f"Ambiguous numeric N-BaIoT labels in {path}: {sorted(values)[:20]}; "
                "provide 0=benign / positive=attack labels or derive labels from filenames."
            )
    else:
        labels = raw.astype(str).map(_norm_label)
        label_values = set(str(v) for v in labels.to_numpy())

        benign_terms = {
            "benign",
            "benign_traffic",
            "normal",
            "normal_traffic",
            "legitimate",
            "0",
            "false",
        }

        if any(v in benign_terms for v in label_values):
            out = (
                ~labels.isin(benign_terms)
            ).to_numpy(dtype=np.int64)
        else:
            attack_re = (
                r"attack|mirai|gafgyt|malicious|anomal|botnet|combo|"
                r"junk|scan|tcp|udp|syn|ack"
            )
            out = labels.str.contains(
                attack_re, regex=True, na=False
            ).to_numpy(dtype=np.int64)

            if int(np.count_nonzero(out)) == 0:
                raise ValueError(
                    f"Cannot infer positive/attack class from N-BaIoT label column in {path}; "
                    f"observed labels={sorted(label_values)[:20]}"
                )

    if out.size == 0 or not bool(np.isin(out, [0, 1]).all()):
        raise ValueError(f"Invalid mapped labels in {path}")

    return np.asarray(out, dtype=np.int64)

def _infer_nbaiot_filename_label(path: Path) -> int:
    """Infer benign/attack from a filename, failing on conflicting evidence."""
    text = str(path).lower()
    benign_tokens = ("benign", "normal_traffic", "normal-traffic", "legitimate")
    attack_tokens = (
        "gafgyt",
        "mirai",
        "attack",
        "malicious",
        "anomal",
        "botnet",
        "combo",
        "junk",
        "scan",
        "tcp",
        "udp",
        "syn",
        "ack",
    )
    is_benign = any(token in text for token in benign_tokens)
    is_attack = any(token in text for token in attack_tokens)
    if is_benign and is_attack:
        raise ValueError(
            f"Ambiguous N-BaIoT filename/path contains both benign and attack markers: {path}. "
            "Rename/place the file so its class is unambiguous or provide an explicit label column."
        )
    if is_benign:
        return 0
    if is_attack:
        return 1
    raise ValueError(
        f"Cannot infer N-BaIoT label from filename/path {path}. "
        "Use filenames/folders containing benign or a known attack name, or provide an explicit label column."
    )


def _read_nbaiot() -> tuple[pd.DataFrame, np.ndarray]:
    """Read N-BaIoT CSVs in chunks and uniformly reservoir-sample rows per file.

    Sampling is bounded by STREAMWEAVE_NBAIOT_ROWS (default 855932), avoids
    materializing whole multi-million-row CSVs, and preserves the source-file
    mixture rather than artificially balancing classes. Within each file, the
    quota rows with the smallest independent uniform random priorities form a
    uniform sample without replacement; their original row order is unspecified.
    N-BaIoT is treated as non-chronological, so within-file order is not used as
    temporal evidence. Headered and headerless schemas are normalized separately;
    mixed representations that do not yield identical normalized feature names
    and order fail closed rather than silently unioning or truncating columns.
    """
    roots = [
        DATA_DIR / "N-BaIoT",
        DATA_DIR / "NBaIoT",
        DATA_DIR / "nbaiot",
        DATA_DIR / "n_baiot",
    ]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files = sorted(
                p
                for p in root.rglob("*.csv")
                if p.is_file() and not p.name.startswith(".")
            )
            if files:
                break
    if not files:
        raise FileNotFoundError(
            f"N-BaIoT CSV files not found. Place the extracted dataset under {DATA_DIR / 'N-BaIoT'} "
            "or set STREAMWEAVE_DATASETS to datasets you have available."
        )

    target_n = int(os.environ.get("STREAMWEAVE_NBAIOT_ROWS", "855932"))
    chunk_size = int(os.environ.get("STREAMWEAVE_NBAIOT_CHUNK_SIZE", "100000"))
    if target_n <= 0 or chunk_size <= 0:
        raise ValueError(
            "STREAMWEAVE_NBAIOT_ROWS and STREAMWEAVE_NBAIOT_CHUNK_SIZE must be positive"
        )

    weights = np.asarray([max(1, p.stat().st_size) for p in files], dtype=np.float64)
    quotas = np.floor(target_n * weights / weights.sum()).astype(int)
    if len(quotas):
        # Distribute rounding remainder without giving one file the entire remainder.
        remainder = target_n - int(quotas.sum())
        for j in range(max(0, remainder)):
            quotas[j % len(quotas)] += 1

    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    reference_columns: list[str] | None = None
    label_keys = {"label", "class", "target", "category", "attack"}

    for i, (path, quota) in enumerate(zip(files, quotas)):
        if quota <= 0:
            continue
        _record_input_file(path)
        headerless = _first_csv_row_is_data(path, min_numeric_fraction=0.80)
        try:
            reader = pd.read_csv(
                path,
                header=None if headerless else "infer",
                low_memory=False,
                chunksize=chunk_size,
            )
            rng = np.random.default_rng(MASTER_SEED + i)
            reservoir_x: np.ndarray | None = None
            reservoir_y = np.empty(0, dtype=np.int64)
            reservoir_keys = np.empty(0, dtype=np.float64)
            rows_seen = 0

            for chunk in reader:
                if chunk.empty:
                    continue
                chunk.columns = [str(c).strip() for c in chunk.columns]
                label_cols = [
                    c for c in chunk.columns if str(c).strip().lower() in label_keys
                ]
                if not label_cols and headerless and chunk.shape[1] == 116:
                    trailing = chunk.columns[-1]
                    # A candidate label must be low-cardinality AND map cleanly
                    # to the documented binary convention; otherwise fail closed.
                    if chunk[trailing].nunique(dropna=True) > 20:
                        raise ValueError(
                            f"Headerless N-BaIoT file {path} has 116 columns with a high-cardinality final field"
                        )
                    validated_trailing_labels = _nbaiot_explicit_labels(
                        chunk[trailing], path
                    )
                    label_cols = [trailing]
                else:
                    validated_trailing_labels = None

                if label_cols:
                    y_chunk = (
                        validated_trailing_labels
                        if validated_trailing_labels is not None
                        and label_cols[0] == chunk.columns[-1]
                        else _nbaiot_explicit_labels(chunk[label_cols[0]], path)
                    )
                    feature_frame = chunk.drop(columns=label_cols).copy()
                else:
                    y_chunk = np.full(
                        len(chunk), _infer_nbaiot_filename_label(path), dtype=np.int64
                    )
                    feature_frame = chunk.copy()

                if headerless:
                    feature_frame.columns = [
                        f"feature_{j}" for j in range(feature_frame.shape[1])
                    ]
                else:
                    feature_frame.columns = [
                        str(c).strip() for c in feature_frame.columns
                    ]
                columns = list(feature_frame.columns)
                if reference_columns is None:
                    reference_columns = columns
                elif columns != reference_columns:
                    raise ValueError(
                        f"N-BaIoT feature schema mismatch in {path}: expected {len(reference_columns)} "
                        f"columns in the same order, found {len(columns)}. Refusing to truncate or union columns."
                    )

                x_chunk = feature_frame.apply(pd.to_numeric, errors="coerce").to_numpy(
                    dtype=np.float32
                )
                if not np.isfinite(x_chunk[~np.isnan(x_chunk)]).all():
                    raise ValueError(f"Non-finite numeric conversion in {path}")
                # Random-priority reservoir: retain the quota rows with smallest
                # independent random keys, which is uniform over all rows in file.
                keys = rng.random(len(x_chunk))
                if reservoir_x is None:
                    candidate_x, candidate_y, candidate_keys = x_chunk, y_chunk, keys
                else:
                    candidate_x = np.concatenate([reservoir_x, x_chunk], axis=0)
                    candidate_y = np.concatenate([reservoir_y, y_chunk], axis=0)
                    candidate_keys = np.concatenate([reservoir_keys, keys], axis=0)
                rows_seen += len(x_chunk)
                if len(candidate_y) > quota:
                    keep = np.argpartition(candidate_keys, quota - 1)[:quota]
                    reservoir_x = candidate_x[keep]
                    reservoir_y = candidate_y[keep]
                    reservoir_keys = candidate_keys[keep]
                else:
                    reservoir_x, reservoir_y, reservoir_keys = (
                        candidate_x,
                        candidate_y,
                        candidate_keys,
                    )
            if reservoir_x is None or len(reservoir_y) == 0:
                continue
            if reservoir_x.shape[1] != 115:
                raise ValueError(
                    f"Expected 115 N-BaIoT features after label removal in {path}, found {reservoir_x.shape[1]}"
                )
            frames.append(pd.DataFrame(reservoir_x, columns=reference_columns))
            labels.append(reservoir_y.astype(np.int64))
            print(
                f"[N-BaIoT] sampled {len(reservoir_y):,}/{rows_seen:,} rows from {path.name}",
                flush=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to stream/sample N-BaIoT file {path}: {exc}"
            ) from exc

    if not frames:
        raise ValueError("No usable N-BaIoT rows were found")
    X = pd.concat(frames, ignore_index=True, verify_integrity=False)
    y = np.concatenate(labels).astype(np.int64)
    if len(X) != len(y):
        raise ValueError("N-BaIoT sampling produced feature/label length mismatch")
    if X.shape[1] != 115:
        raise ValueError(
            f"Expected 115 N-BaIoT features after label removal, found {X.shape[1]}"
        )
    if set(np.unique(y)) != {0, 1}:
        counts = dict(zip(*np.unique(y, return_counts=True)))
        raise ValueError(
            f"N-BaIoT labels must contain both benign=0 and attack=1 after sampling; got {counts}"
        )
    return X, y


def _openml(data_id: int) -> Any:
    """Fetch a pinned OpenML dataset ID into the configured local cache."""
    try:
        return fetch_openml(
            data_id=data_id, as_frame=True, parser="auto", data_home=str(DATA_DIR)
        )
    except TypeError:
        return fetch_openml(
            data_id=data_id, as_frame=True, data_home=str(DATA_DIR)
        )


def _local_candidates(name: str) -> list[Path]:
    """Find local dataset downloads using exact names/stems, not broad prefixes."""
    aliases = {
        "Adult": {"phpmawtba", "adult", "adult.data", "adult_data"},
        "BankMarketing": {
            "phpkixskf",
            "bank-marketing",
            "bank_full",
            "bank-full",
            "bank",
        },
        "Electricity": {"electricity-normalized", "electricity", "2419"},
    }
    valid_suffixes = {"", ".csv", ".arff", ".data", ".txt", ".zip"}
    stems = aliases.get(name, set())
    roots = [DATA_DIR, BASE_DIR, Path.cwd()]
    found: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        # Search one level and recursively for the exact stem, but reject unrelated
        # names such as electricity_something.csv.
        for q in root.rglob("*"):
            if not q.is_file():
                continue
            if q.suffix.lower() not in valid_suffixes:
                continue
            stem = q.stem.lower()
            full = q.name.lower()
            matches = stem in stems or full in stems
            if not matches:
                continue
            key = str(q.resolve())
            if key not in seen:
                seen.add(key)
                found.append(q)
    # Prefer obvious exact ARFF/CSV downloads and stable order.
    priority = {".arff": 0, ".csv": 1, "": 2, ".data": 3, ".txt": 4, ".zip": 5}
    return sorted(
        found, key=lambda q: (priority.get(q.suffix.lower(), 10), len(str(q)), str(q))
    )


def _decode_arff_frame(data) -> pd.DataFrame:
    df = pd.DataFrame(data)
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(
                lambda v: (
                    v.decode("utf-8", errors="replace")
                    if isinstance(v, (bytes, bytearray))
                    else v
                )
            )
    return df


def _read_local_table(path: Path, dataset: str) -> pd.DataFrame:
    """Read OpenML ARFF (including extensionless downloads) or CSV data."""
    with path.open("rb") as f:
        prefix = f.read(32768)
    looks_arff = (
        b"@relation" in prefix.lower()
        and b"@attribute" in prefix.lower()
        and b"@data" in prefix.lower()
    )
    if path.suffix.lower() == ".arff" or looks_arff:
        data, _ = scipy_arff.loadarff(str(path))
        return _decode_arff_frame(data)
    if dataset == "Adult":
        adult_cols = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "class",
        ]
        # UCI adult.data is headerless; a named CSV/ARFF-derived CSV is detected separately.
        first = (
            prefix.decode("utf-8", errors="replace").splitlines()[0] if prefix else ""
        )
        if not any(
            token in first.lower() for token in ("age", "class", "income", "target")
        ):
            df = pd.read_csv(
                path, header=None, skipinitialspace=True, na_values=["?", " ?"]
            )
            if df.shape[1] == 15:
                df.columns = adult_cols
            return df
    if dataset == "BankMarketing":
        first = (
            prefix.decode("utf-8", errors="replace").splitlines()[0] if prefix else ""
        )
        sep = ";" if first.count(";") > first.count(",") else None
        return pd.read_csv(path, sep=sep, engine="python", skipinitialspace=True)
    return pd.read_csv(path, sep=None, engine="python", skipinitialspace=True)


def _load_local_openml_dataset(
    name: str,
) -> tuple[pd.DataFrame, np.ndarray, bool, str] | None:
    """Prefer the exact files the user downloaded; return None when absent."""
    paths = _local_candidates(name)
    if not paths:
        return None
    candidate_failures: list[str] = []
    for path in paths:
        try:
            _record_input_file(path)
            df = _read_local_table(path, name)
            if df.empty:
                candidate_failures.append(
                    f"{path}: parsed successfully but contains no rows"
                )
                continue
            if df.shape[1] < 2:
                candidate_failures.append(
                    f"{path}: parsed to {df.shape[0]} row(s) and {df.shape[1]} column(s); "
                    "at least two columns are required"
                )
                continue
            lower = {str(c).strip().lower(): c for c in df.columns}
            if name == "Adult":
                target = next(
                    (
                        lower[k]
                        for k in ("class", "income", "target", "label", "y")
                        if k in lower
                    ),
                    df.columns[-1],
                )
                y = _binary_labels(df[target], positive=">50k")
                X = df.drop(columns=[target])
                return (
                    X.reset_index(drop=True),
                    y,
                    False,
                    "binary_classification_not_intrinsically_anomaly_detection",
                )
            if name == "BankMarketing":
                target = next(
                    (
                        lower[k]
                        for k in ("y", "class", "target", "label", "subscribed")
                        if k in lower
                    ),
                    df.columns[-1],
                )
                y = _binary_labels(df[target], positive="yes")
                X = df.drop(columns=[target])
                # The common Bank Marketing files contain month/day fields but no
                # reliable full timestamp. Do not claim row order is chronological.
                return (
                    X.reset_index(drop=True),
                    y,
                    False,
                    "binary_classification_not_intrinsically_anomaly_detection",
                )
            if name == "Electricity":
                target = next(
                    (lower[k] for k in ("class", "target", "label", "y") if k in lower),
                    df.columns[-1],
                )
                y = _binary_labels(df[target], positive="up")
                X = df.drop(columns=[target])
                # OpenML's normalized ARFF date column is an ordinal-like
                # normalized field; preserve source row order and exclude it as a predictor.
                date_cols = [
                    c
                    for c in X.columns
                    if str(c).strip().lower() in {"date", "timestamp", "datetime"}
                ]
                if date_cols:
                    X = X.drop(columns=date_cols)
                return (
                    X.reset_index(drop=True),
                    y,
                    True,
                    "chronological_concept_drift_classification_not_anomaly_detection",
                )
        except Exception as exc:
            candidate_failures.append(f"{path}: {type(exc).__name__}: {exc}")
    # `None` is reserved for the genuinely absent-local-data case. If candidates
    # exist but are malformed/unusable, fail closed rather than silently causing
    # load_dataset() to fetch a different copy from the network.
    details = (
        "\n  - ".join(candidate_failures) or "no candidate produced a usable table"
    )
    raise RuntimeError(
        f"Found local candidate file(s) for {name}, but none was usable. "
        "Correct/remove the listed file(s) or explicitly choose a different data source:\n  - "
        f"{details}"
    )


def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, bool, str]:
    """Return raw X, binary y, chronological flag, and task description."""
    if name in {"Adult", "BankMarketing", "Electricity"}:
        local = _load_local_openml_dataset(name)
        if local is not None:
            return local
    if name == "Covertype":
        ds: Any = fetch_covtype(data_home=str(DATA_DIR))
        return (
            pd.DataFrame(
                ds.data, columns=[f"feature_{i}" for i in range(ds.data.shape[1])]
            ),
            (np.asarray(ds.target) >= 3).astype(np.int64),
            False,
            "anomaly_detection",
        )
    if name == "Skin":
        X, y = _read_skin()
        return X, y, False, "anomaly_detection_proxy_non_skin_positive"
    if name == "N-BaIoT":
        X, y = _read_nbaiot()
        return X, y, False, "iot_botnet_intrusion_detection"
    if name == "NSL-KDD":
        X, y = _read_nsl_kdd()
        return X, y, False, "intrusion_detection"
    if name == "CreditCardFraud":
        ds = _openml(1597)
        X = ds.data.copy()
        y = _binary_labels(ds.target, positive="1")
        chrono = False
        if "Time" in X.columns:
            

            time_values = np.asarray(
                pd.to_numeric(X["Time"], errors="coerce"),
                dtype=np.float64,
            )

            time_values = np.where(np.isnan(time_values), -np.inf, time_values)

            order = np.argsort(time_values, kind="stable")

            X, y = X.iloc[order].reset_index(drop=True), y[order]
            X = X.drop(columns=["Time"])
            chrono = True
        return X.reset_index(drop=True), y, chrono, "fraud_detection"
    if name == "Adult":
        ds = _openml(1590)
        X = ds.data.copy()
        y = _binary_labels(ds.target, positive=">50k")
        return (
            X.reset_index(drop=True),
            y,
            False,
            "binary_classification_not_intrinsically_anomaly_detection",
        )
    if name == "BankMarketing":
        ds = _openml(1461)
        X = ds.data.copy()
        y = _binary_labels(ds.target, positive="yes")
        # No full timestamp is available in this source; treat it as non-temporal.
        return (
            X.reset_index(drop=True),
            y,
            False,
            "binary_classification_not_intrinsically_anomaly_detection",
        )
    if name == "Electricity":
        ds = _openml(151)
        X = ds.data.copy()
        y = _binary_labels(ds.target, positive="up")
        chrono = True
        # Prefer real date/period order where present; otherwise retain the
        # documented original Elec2 record order. Date is used only for ordering.
        X = X.reset_index(drop=True)
        date_cols = [
            c
            for c in X.columns
            if str(c).strip().lower() in {"date", "timestamp", "datetime"}
        ]
        if date_cols:
            dcol = date_cols[0]
            dates = pd.to_datetime(X[dcol], errors="coerce")
            if dates.notna().all():
                order = np.argsort(dates.to_numpy(), kind="stable")
                X = X.iloc[order].reset_index(drop=True)
                y = y[order]
            X = X.drop(columns=[dcol])
        return (
            X.reset_index(drop=True),
            y,
            chrono,
            "chronological_concept_drift_classification_not_anomaly_detection",
        )
    raise ValueError(
        f"Unknown dataset {name!r}. Supported: Covertype, Skin, N-BaIoT, NSL-KDD, CreditCardFraud, Adult, BankMarketing, Electricity"
    )


# ----------------------- Split and preprocessing ---------------------------
@dataclass
class DataSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    X_cal_fit: np.ndarray
    y_cal_fit: np.ndarray
    X_cal_thresh: np.ndarray
    y_cal_thresh: np.ndarray
    X_adapt: np.ndarray
    y_adapt: np.ndarray
    X_stream: np.ndarray
    y_stream: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    chronological: bool
    task: str
    feature_names: list[str]
    dataset_fingerprint: str


def _split_indices(
    y: np.ndarray, chronological: bool, seed: int
) -> dict[str, np.ndarray]:
    n = len(y)
    if n < 200:
        raise ValueError(f"Dataset has only {n} rows; too small for this protocol")
    names = list(FRACTIONS)
    if not math.isclose(sum(FRACTIONS.values()), 1.0, abs_tol=1e-9):
        raise ValueError("FRACTIONS must sum to 1.0")
    if chronological:
        cuts = np.rint(np.cumsum([FRACTIONS[k] for k in names]) * n).astype(int)
        starts = np.r_[0, cuts[:-1]]
        return {
            k: np.arange(starts[i], cuts[i], dtype=int) for i, k in enumerate(names)
        }
    rng = seed
    remaining = np.arange(n)
    out = {}
    remaining_frac = 1.0
    for i, key in enumerate(names[:-1]):
        frac = FRACTIONS[key]
        rel = frac / remaining_frac
        strat = (
            y[remaining]
            if len(np.unique(y[remaining])) == 2
            and np.min(np.bincount(y[remaining])) >= 2
            else None
        )
        chosen, remaining = train_test_split(
            remaining, train_size=rel, random_state=rng, stratify=strat
        )
        out[key] = np.asarray(chosen, dtype=int)
        remaining_frac -= frac
        rng += 17
    out[names[-1]] = np.asarray(remaining, dtype=int)
    return out


def _make_ohe() -> OneHotEncoder:
    try:
        encoder: OneHotEncoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore")
    return encoder


def preprocess_split(
    X: pd.DataFrame, y: np.ndarray, chronological: bool, task: str, seed: int
) -> DataSplit:
    X = X.copy().reset_index(drop=True)
    y = np.asarray(y, dtype=np.int64)
    if len(X) != len(y):
        raise ValueError("Feature/label row count mismatch")
    if len(np.unique(y)) != 2 or not np.isin(y, [0, 1]).all():
        counts = dict(zip(*np.unique(y, return_counts=True)))
        raise ValueError(
            f"Expected binary labels 0/1 with both classes present; got {counts}"
        )
    # Normalize missing/infinite raw numeric values before determining columns.
    cat_cols = [
        c
        for c in X.columns
        if (
            pd.api.types.is_object_dtype(X[c].dtype)
            or pd.api.types.is_string_dtype(X[c].dtype)
            or isinstance(X[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(X[c].dtype)
        )
    ]
    num_cols = [c for c in X.columns if c not in cat_cols]
    raw_num = (
        X[num_cols].apply(pd.to_numeric, errors="coerce")
        if num_cols
        else pd.DataFrame(index=X.index)
    )
    if num_cols:
        raw_num = raw_num.replace([np.inf, -np.inf], np.nan)
    raw_cat = X[cat_cols].copy() if cat_cols else pd.DataFrame(index=X.index)
    raw_cat = raw_cat.fillna("__MISSING__").astype(str) if cat_cols else raw_cat

    ids = _split_indices(y, chronological, seed)
    train_ids = ids["train"]
    medians = (
        raw_num.iloc[train_ids].median(numeric_only=True)
        if num_cols
        else pd.Series(dtype=float)
    )
    if num_cols:
        medians = medians.fillna(0.0)
        num_arr = raw_num.fillna(medians).to_numpy(dtype=np.float32)
        scaler = StandardScaler()
        scaler.fit(num_arr[train_ids])
        scaled: Any = scaler.transform(num_arr)
        num_arr = np.asarray(scaled, dtype=np.float32)
    else:
        scaler = None
        num_arr = np.empty((len(X), 0), dtype=np.float32)
    
    if cat_cols:
        enc = _make_ohe()
        enc.fit(raw_cat.iloc[train_ids])

        encoded = enc.transform(raw_cat)
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()

        cat_arr: np.ndarray = np.asarray(encoded, dtype=np.float32)
        cat_names = enc.get_feature_names_out(cat_cols).tolist()

    else:
        enc = None
        cat_arr = np.empty((len(X), 0), dtype=np.float32)
        cat_names = []
    all_arr = np.hstack([num_arr, cat_arr]).astype(np.float32)
    if not np.isfinite(all_arr).all():
        raise ValueError("Preprocessing produced NaN or infinite values")
    feature_names = [str(c) for c in num_cols] + cat_names
    fingerprint_schema = json.dumps(
        {
            "shape": list(X.shape),
            "columns": [str(c) for c in X.columns],
            "dtypes": [str(dtype) for dtype in X.dtypes],
            "task": str(task),
            "chronological": bool(chronological),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    

    fingerprint = hashlib.sha256(
        fingerprint_schema
        + X.to_csv(index=True, lineterminator="\n").encode("utf-8")
        + y.tobytes()
    ).hexdigest()

    idx = ids
    return DataSplit(
        X_train=all_arr[idx["train"]],
        y_train=y[idx["train"]],
        X_cal_fit=all_arr[idx["cal_fit"]],
        y_cal_fit=y[idx["cal_fit"]],
        X_cal_thresh=all_arr[idx["cal_thresh"]],
        y_cal_thresh=y[idx["cal_thresh"]],
        X_adapt=all_arr[idx["adapt"]],
        y_adapt=y[idx["adapt"]],
        X_stream=all_arr[idx["stream"]],
        y_stream=y[idx["stream"]],
        X_test=all_arr[idx["test"]],
        y_test=y[idx["test"]],
        chronological=chronological,
        task=task,
        feature_names=feature_names,
        dataset_fingerprint=fingerprint,
    )


def sample_train(
    X: np.ndarray, y: np.ndarray, cap: int | None, seed: int, chronological: bool
) -> tuple[np.ndarray, np.ndarray]:
    if cap is None or cap <= 0 or len(y) <= cap:
        return X, y
    if chronological:
        # Use a deterministic evenly spaced subset so that the entire training
        # period is represented without altering the subsequent stream order.
        idx = np.linspace(0, len(y) - 1, num=cap, dtype=int)
        return X[idx], y[idx]
    idx = np.arange(len(y))
    try:
        keep, _ = train_test_split(idx, train_size=cap, random_state=seed, stratify=y)
    except ValueError:
        rng = np.random.default_rng(seed)
        keep = rng.choice(idx, size=cap, replace=False)
    keep = np.sort(keep)
    return X[keep], y[keep]


def subset_for_scoring(
    X: np.ndarray, y: np.ndarray, cap: int, seed: int, chronological: bool
) -> tuple[np.ndarray, np.ndarray]:
    if cap <= 0 or len(y) <= cap:
        return X, y
    if chronological:
        idx = np.linspace(0, len(y) - 1, num=cap, dtype=int)
    else:
        all_idx = np.arange(len(y))
        try:
            idx, _ = train_test_split(
                all_idx, train_size=cap, random_state=seed, stratify=y
            )
        except ValueError:
            idx = np.random.default_rng(seed).choice(all_idx, size=cap, replace=False)
        idx = np.sort(idx)
    return X[idx], y[idx]


# --------------------------------- Metrics ----------------------------------
def safe_pr_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Average precision; return NaN only for a valid but single-class window.

    Shape, label, and non-finite-score errors are deliberately not swallowed.
    """
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError(f"PR-AUC length mismatch: {len(y)} labels vs {len(p)} scores")
    if not np.isfinite(p).all():
        raise ValueError("PR-AUC scores contain NaN or infinity")
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("PR-AUC labels must be binary 0/1")
    return float(average_precision_score(y, np.clip(p, 0.0, 1.0)))


def metric_dict(
    y: np.ndarray, p: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    if len(y) != len(p):
        raise ValueError(
            f"Metric length mismatch: {len(y)} labels vs {len(p)} predictions"
        )
    if not np.isfinite(p).all():
        raise ValueError("Non-finite predictions")
    p = np.clip(p, 0.0, 1.0)
    pred = p >= float(threshold)
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float("nan")
    try:
        ll = float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))
    except Exception:
        ll = float("nan")
    return {
        "acc": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division="warn")),
        "f1_fixed_05": float(f1_score(y, p >= 0.5, zero_division="warn")),
        "roc_auc": auc,
        "pr_auc": safe_pr_auc(y, p),
        "logloss": ll,
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
        "threshold": float(threshold),
    }


def tune_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y) < 2 or len(np.unique(y)) < 2 or not np.isfinite(p).all():
        return 0.5
    # Search observed-score quantiles plus conventional fallback; no arbitrary
    # 0.10-0.90 restriction, and no test/stream labels enter this step.
    candidates = np.unique(
        np.r_[np.quantile(p, np.linspace(0.0, 1.0, 201)), 0.5, 0.0, 1.0]
    )
    best_t, best_f = 0.5, -1.0
    for t in candidates:
        f = f1_score(y, p >= t, zero_division="warn")
        if f > best_f or (f == best_f and abs(t - 0.5) < abs(best_t - 0.5)):
            best_t, best_f = float(t), float(f)
    return best_t


# --------------------------------- Models -----------------------------------
class TabularNet(nn.Module):
    def __init__(self, kind: str, d: int):
        super().__init__()
        self.kind = kind
        if kind == "MLP":
            self.encoder = nn.Sequential(
                nn.Linear(d, 128),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            zdim = 64
        elif kind == "SmallMLP":
            self.encoder = nn.Sequential(
                nn.Linear(d, 48), nn.ReLU(), nn.Linear(48, 24), nn.ReLU()
            )
            zdim = 24
        elif kind == "WideMLP":
            self.encoder = nn.Sequential(
                nn.Linear(d, 256),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            zdim = 128
        elif kind == "DeepAE":
            h, zdim = min(128, max(32, d * 2)), min(64, max(16, d))
            self.encoder = nn.Sequential(
                nn.Linear(d, h), nn.ReLU(), nn.Linear(h, zdim), nn.ReLU()
            )
            self.decoder = nn.Sequential(nn.Linear(zdim, h), nn.ReLU(), nn.Linear(h, d))
        elif kind == "ResidualMLP":
            h = 128
            self.input_layer = nn.Sequential(
                nn.Linear(d, h), nn.LayerNorm(h), nn.GELU()
            )
            self.residual = nn.Sequential(
                nn.Linear(h, h), nn.GELU(), nn.Dropout(0.10), nn.Linear(h, h)
            )
            self.norm = nn.LayerNorm(h)
            self.encoder = nn.Sequential(nn.Linear(h, 64), nn.GELU())
            zdim = 64
        elif kind == "FeatureAttention":
            h = 32
            self.value_projection = nn.Linear(1, h)
            self.feature_embedding = nn.Parameter(torch.randn(d, h) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=h,
                nhead=4,
                dim_feedforward=64,
                dropout=0.10,
                batch_first=True,
                norm_first=True,
            )
            self.attention = nn.TransformerEncoder(
                layer, num_layers=1, enable_nested_tensor=False
            )
            self.encoder = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, 32), nn.GELU())
            zdim = 32
        elif kind == "GRU":
            # Feature-wise GRU baseline: one row's ordered feature vector is
            # treated as a sequence of scalar feature tokens. This is NOT a
            # temporal-event GRU; report it as a feature-sequence architecture.
            self.gru = nn.GRU(
                input_size=1, hidden_size=32, num_layers=1, batch_first=True
            )
            self.encoder = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU())
            zdim = 32
        elif kind == "TCN":
            # Feature-wise temporal-convolution analogue; convolutions run over
            # the feature axis, not over past events in the stream.
            self.tcn = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=2, dilation=2),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.encoder = nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 32), nn.GELU())
            zdim = 32
        else:
            raise ValueError(f"Unknown model architecture {kind}")
        self.head = nn.Linear(zdim, 1)
        self.input_dim = d

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "ResidualMLP":
            h = self.input_layer(x)
            h = F.gelu(self.norm(h + self.residual(h)))
            return self.encoder(h)
        if self.kind == "FeatureAttention":
            if x.shape[1] != self.feature_embedding.shape[0]:
                raise ValueError("FeatureAttention input dimension changed")
            tokens = self.value_projection(
                x.unsqueeze(-1)
            ) + self.feature_embedding.unsqueeze(0)
            z = self.attention(tokens).mean(dim=1)
            return self.encoder(z)
        if self.kind == "GRU":
            z = self.gru(x.unsqueeze(-1))[0][:, -1, :]
            return self.encoder(z)
        if self.kind == "TCN":
            z = self.tcn(x.unsqueeze(1)).squeeze(-1)
            return self.encoder(z)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x)).squeeze(-1)

    def reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind != "DeepAE":
            raise AttributeError("reconstruction is available only for DeepAE")
        return self.decoder(self.forward_features(x))


def make_model(kind: str, d: int) -> TabularNet:
    return TabularNet(kind, d).to(DEVICE)


def _pos_weight(y: np.ndarray) -> float:
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return 1.0
    return float(np.clip(neg / max(pos, 1), 1.0, max(1.0, POS_WEIGHT_CAP)))


def train_classifier(
    model: TabularNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed: int,
    lr: float = BASE_LR,
    batch_size: int = 256,
    reconstruction: bool = True,
) -> TabularNet:
    if len(y) == 0:
        raise ValueError("Cannot train on an empty partition")
    seed_everything(seed)
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    pw = _pos_weight(y.astype(int))
    object.__setattr__(model, "_streamweave_pos_weight", pw)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(
        ds,
        batch_size=min(batch_size, len(ds)),
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32, device=DEVICE)
    )
    model.train()
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            if reconstruction and model.kind == "DeepAE":
                loss = loss + 0.10 * F.mse_loss(model.reconstruction(xb), xb)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    return model


@torch.no_grad()
def predict_proba(
    model: TabularNet, X: np.ndarray, batch_size: int = 1024
) -> np.ndarray:
    model.eval()
    result = []
    for st in range(0, len(X), batch_size):
        xb = torch.as_tensor(
            np.asarray(X[st : st + batch_size], dtype=np.float32), device=DEVICE
        )
        result.append(torch.sigmoid(model(xb)).detach().cpu().numpy())
    return np.concatenate(result) if result else np.empty(0, dtype=np.float32)


@torch.no_grad()
def _features(model: TabularNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    chunks = []
    for st in range(0, len(X), 1024):
        xb = torch.as_tensor(
            np.asarray(X[st : st + 1024], dtype=np.float32), device=DEVICE
        )
        chunks.append(model.forward_features(xb).detach().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 1), np.float32)


def _teacher_indices(
    n: int, n_models: int, chronological: bool, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if chronological:
        # Overlapping rolling windows emulate successive BPU snapshots without
        # moving any later stream/test records into training.
        win = min(n, max(512, int(0.70 * n)))
        starts = np.linspace(0, max(0, n - win), num=n_models, dtype=int)
        return [idx[s : s + win] for s in starts]
    out = []
    size = min(n, max(256, int(0.85 * n)))
    for _ in range(n_models):
        out.append(rng.choice(idx, size=size, replace=False))
    return out


def train_bpu_pool(
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    seed: int,
    chronological: bool,
    n_models: int = BPU_HISTORY_SIZE,
) -> list[TabularNet]:
    pool = []
    for i, idx in enumerate(
        _teacher_indices(len(y), n_models, chronological, seed + 97)
    ):
        model = make_model(kind, X.shape[1])
        pool.append(
            train_classifier(
                model,
                X[idx],
                y[idx],
                epochs,
                seed + 1009 * (i + 1) + seed,
                reconstruction=True,
            )
        )
    return pool


def _kd_epoch(
    student: TabularNet,
    teachers: list[TabularNet],
    X: np.ndarray,
    y: np.ndarray,
    method: str,
    epochs: int,
    seed: int,
) -> TabularNet:
    """Apply one of the explicitly documented, lightweight neural transfer rules."""
    if method == "None" or len(y) == 0:
        return student
    seed_everything(seed)
    for t in teachers:
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
    student.train()
    ds = TensorDataset(
        torch.from_numpy(np.asarray(X, np.float32)),
        torch.from_numpy(np.asarray(y, np.float32)),
    )
    loader = DataLoader(
        ds, batch_size=min(128, len(ds)), shuffle=True, num_workers=NUM_WORKERS
    )
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY
    )
    pw = float(getattr(student, "_streamweave_pos_weight", _pos_weight(y.astype(int))))
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32, device=DEVICE)
    )
    projection: nn.Linear | None = None
    contrast_t: nn.Linear | None = None
    contrast_s: nn.Linear | None = None
    opt_params = list(student.parameters())
    if method == "FitNets":
        teacher_dim = _features(teachers[-1], X[: min(2, len(X))]).shape[1]
        student_dim = _features(student, X[: min(2, len(X))]).shape[1]
        projection = nn.Linear(student_dim, teacher_dim).to(DEVICE)
        opt_params += list(projection.parameters())
        optimizer = torch.optim.AdamW(opt_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    elif method == "ContrastiveRepKD":
        teacher_dim = _features(teachers[-1], X[: min(2, len(X))]).shape[1]
        student_dim = _features(student, X[: min(2, len(X))]).shape[1]
        contrast_t = nn.Linear(teacher_dim, 64, bias=False).to(DEVICE)
        contrast_s = nn.Linear(student_dim, 64, bias=False).to(DEVICE)
        # Keep the teacher backbone frozen, but learn both projection heads.
        # A randomly initialized, permanently frozen teacher projection would
        # make the contrastive target arbitrary and weaken the KD comparison.
        opt_params += list(contrast_t.parameters()) + list(contrast_s.parameters())
        optimizer = torch.optim.AdamW(opt_params, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    # _features() switches any model it touches to eval() mode. Restore the
    # student's training mode after dimension probing for FitNets/ContrastiveRepKD.
    student.train()
    temperature = 2.0
    alpha = 0.35 if method == "BornAgainStyleKD" else 0.5
    for _ in range(max(1, epochs)):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            slog = student(xb)
            hard = bce(slog, yb)
            if method in {"StandardKD", "BornAgainStyleKD", "EnsembleDistill"}:
                with torch.no_grad():
                    tprob_stack = torch.stack(
                        [
                            torch.sigmoid(t(xb) / temperature)
                            for t in (
                                teachers
                                if method == "EnsembleDistill"
                                else [teachers[-1]]
                            )
                        ],
                        dim=0,
                    )
                    tprob = tprob_stack.mean(dim=0).clamp(1e-5, 1 - 1e-5)
                soft = (
                    F.binary_cross_entropy(
                        torch.sigmoid(slog / temperature).clamp(1e-5, 1 - 1e-5), tprob
                    )
                    * temperature**2
                )
                loss = alpha * soft + (1 - alpha) * hard
            elif method == "RKD":
                with torch.no_grad():
                    tz = F.normalize(teachers[-1].forward_features(xb), dim=1)
                    td = torch.cdist(tz, tz)
                sz = F.normalize(student.forward_features(xb), dim=1)
                sd = torch.cdist(sz, sz)
                mask = ~torch.eye(len(xb), dtype=torch.bool, device=DEVICE)
                tdv, sdv = td[mask], sd[mask]
                if tdv.numel():
                    tdv = tdv / (tdv.mean().detach() + 1e-6)
                    sdv = sdv / (sdv.mean().detach() + 1e-6)
                    relational = F.smooth_l1_loss(sdv, tdv)
                else:
                    relational = torch.zeros((), device=DEVICE)
                loss = hard + 0.5 * relational
            elif method == "FitNets":
                if projection is None:
                    raise RuntimeError("FitNets projection was not initialized")
                with torch.no_grad():
                    tf = teachers[-1].forward_features(xb)
                sf = student.forward_features(xb)
                loss = hard + 0.5 * F.mse_loss(projection(sf), tf)
            elif method == "ContrastiveRepKD":
                if contrast_t is None or contrast_s is None:
                    raise RuntimeError(
                        "ContrastiveRepKD projection heads were not initialized"
                    )
                with torch.no_grad():
                    tf = teachers[-1].forward_features(xb)
                    tp = F.normalize(contrast_t(tf), dim=1)
                sp = F.normalize(contrast_s(student.forward_features(xb)), dim=1)
                if len(xb) > 1:
                    logits = (sp @ tp.T) / 0.2
                    labels = torch.arange(len(xb), device=DEVICE)
                    contrastive = F.cross_entropy(logits, labels)
                else:
                    # InfoNCE has no negatives for a singleton batch; retain the
                    # supervised objective rather than adding a meaningless zero.
                    contrastive = torch.zeros((), device=DEVICE)
                loss = hard + 0.5 * contrastive
            else:
                raise ValueError(f"Unknown transfer method {method}")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss for transfer={method}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 5.0)
            optimizer.step()
    return student


def _train_student_with_matched_budget(
    student: TabularNet,
    teachers: list[TabularNet],
    X: np.ndarray,
    y: np.ndarray,
    transfer: str,
    extra_epochs: int,
    seed: int,
) -> TabularNet:
    """Match added optimizer epochs between KD methods and the no-KD control.

    For transfer="None", the additional budget uses supervised BCE on the same
    training partition. KD methods use their respective transfer objective for
    the same number of epochs. This controls epoch count, not wall-clock time.
    """
    if extra_epochs <= 0:
        return student
    if transfer == "None":
        return train_classifier(student, X, y, extra_epochs, seed, reconstruction=False)
    return _kd_epoch(student, teachers, X, y, transfer, extra_epochs, seed)


def _reset_online_optimizer(model: TabularNet) -> None:
    if hasattr(model, "_online_optimizer"):
        delattr(model, "_online_optimizer")


def online_update(
    model: TabularNet, X: np.ndarray, y: np.ndarray, epochs: int = ONLINE_EPOCHS
) -> None:
    """Update only after labels are revealed, with class weights from a recent-label window."""
    model.train()
    optimizer = getattr(model, "_online_optimizer", None)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=ONLINE_LR)
        object.__setattr__(model, "_online_optimizer", optimizer)
    xb = torch.as_tensor(np.asarray(X, np.float32), device=DEVICE)
    y_np = np.asarray(y, dtype=np.int64).reshape(-1)
    yb = torch.as_tensor(y_np.astype(np.float32), device=DEVICE)

    prior = np.asarray(
        getattr(model, "_streamweave_online_labels", np.empty(0, dtype=np.int64)),
        dtype=np.int64,
    )
    history = np.concatenate([prior, y_np])[-max(1, MEMORY_MAX) :]
    object.__setattr__(model, "_streamweave_online_labels", history)
    # Do not retain the original training partition's class weight indefinitely:
    # in a naturally changing stream, recent observed prevalence may differ.
    pw = _pos_weight(history)
    object.__setattr__(model, "_streamweave_pos_weight_online", float(pw))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pw, dtype=torch.float32, device=DEVICE)
    )
    for _ in range(max(1, epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite online update loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()


# -------------------------- Combination units -------------------------------
def fit_cu(name: str, P: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Fit a combination unit from model probabilities on CAL_FIT only."""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=int)
    if P.ndim != 2 or P.shape[0] != len(y):
        raise ValueError("Invalid probability matrix for CU fit")
    n_models = P.shape[1]
    params: dict[str, Any] = {"name": name, "n_models": n_models}
    if name == "HDT_Analog":
        # Paper-inspired neural analogue: equal vote of up to five recent batch
        # models plus the active stream model. Not the canonical decision-tree HDT.
        params["mode"] = "all_equal"
    elif name == "SimpleAvg":
        # A simple two-model reference: newest BPU + active SPU.
        params["mode"] = "newest_bpu_spu"
    elif name in {"ConfWeight", "Bayesian", "ErrorWeighted"}:
        eps = 1e-7
        if name == "ConfWeight":
            err = np.mean((P - y[:, None]) ** 2, axis=0)
            scores = -err / 0.05
        elif name == "Bayesian":
            clipped = np.clip(P, eps, 1 - eps)
            scores = np.mean(
                y[:, None] * np.log(clipped) + (1 - y[:, None]) * np.log(1 - clipped),
                axis=0,
            )
        else:
            err = np.mean(np.abs(P - y[:, None]), axis=0)
            scores = -err / 0.05
        scores = scores - np.max(scores)
        weights = np.exp(np.clip(scores, -50, 0))
        weights /= weights.sum()
        params["weights"] = weights.tolist()
    elif name == "Stacking":
        if len(np.unique(y)) == 2:
            meta = LogisticRegression(
                max_iter=500, random_state=0, class_weight="balanced"
            )
            meta.fit(P, y)
            # sklearn's stubs type coef_/intercept_ loosely (sometimes float,
            # sometimes ndarray). Normalize to ndarray before indexing so the
            # static check sees a well-typed 1-D array.
            coef = np.asarray(meta.coef_, dtype=float).reshape(-1)
            intercept = np.asarray(meta.intercept_, dtype=float).reshape(-1)
            params["meta_coef"] = coef.tolist()
            params["meta_intercept"] = float(intercept[0])
        else:
            params["meta_coef"] = None
    elif name == "TempScale":
        # Conventional temperature scaling of mean logits with temperature > 0.
        eps = 1e-7

        logits = np.log(np.clip(P, eps, 1 - eps) / (1 - np.clip(P, eps, 1 - eps))).mean(
            axis=1
        )

        def objective(log_t: float) -> float:
            t = math.exp(float(log_t))
            z = logits / t
            return float(np.mean(np.logaddexp(0, z) - y * z))

        # scipy's stubs expose minimize_scalar()'s result weakly (Pyright infers
        # `object`), so label the OptimizeResult as Any before touching `.x`.
        opt_result: Any = minimize_scalar(
            objective, bounds=(math.log(0.05), math.log(20.0)), method="bounded"
        )
        params["temperature"] = float(math.exp(float(opt_result.x)))
    else:
        raise ValueError(f"Unknown CU {name}")
    return params


def predict_cu(params: dict[str, Any], P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64)
    name = params["name"]
    if P.ndim != 2 or P.shape[1] != params["n_models"]:
        raise ValueError("CU/model-count mismatch")
    eps = 1e-7
    if name == "HDT_Analog":
        out = P.mean(axis=1)
    elif name == "SimpleAvg":
        out = P[:, -2:].mean(axis=1)
    elif name in {"ConfWeight", "Bayesian", "ErrorWeighted"}:
        out = P @ np.asarray(params["weights"], dtype=float)
    elif name == "Stacking":
        if params.get("meta_coef") is None:
            out = P.mean(axis=1)
        else:
            z = P @ np.asarray(params["meta_coef"], dtype=float) + float(
                params["meta_intercept"]
            )
            out = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    elif name == "TempScale":
        logits = np.log(np.clip(P, eps, 1 - eps) / (1 - np.clip(P, eps, 1 - eps))).mean(
            axis=1
        )
        z = logits / float(params["temperature"])
        out = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
    else:
        raise ValueError(name)
    return np.clip(np.asarray(out, dtype=np.float32), 0.0, 1.0)


def _predict_matrix(models: list[TabularNet], X: np.ndarray) -> np.ndarray:
    # The last model is always the active SPU; BPU pool order is oldest -> newest.
    return np.stack([predict_proba(m, X) for m in models], axis=1)


# ------------------- Cache-efficient budgeted pairwise screen -------------------
# --------------------------- Prequential pipeline ---------------------------
def _effective_batch_interval(n_stream: int) -> int:
    """Choose a refresh interval that can affect later predictions on short streams.

    A refresh occurs only at the end of a scored batch and never after the final
    batch. For streams with fewer than two batches, refresh is unnecessary.
    """
    if BATCH_INTERVAL <= 0 or n_stream <= STREAM_BATCH_SIZE:
        return 0
    # Aim for roughly three opportunities across longer streams while honoring
    # the configured maximum. Actual refreshes remain bounded by batch boundaries.
    return max(1, min(int(BATCH_INTERVAL), max(1, int(n_stream) // 3)))


def run_prequential(
    split: DataSplit,
    bpu_kind: str,
    spu_kind: str,
    transfer: str,
    cu_name: str,
    train_size: int,
    epochs: int,
    kd_epochs: int,
    seed: int,
    run_test: bool = False,
    refresh_enabled: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray | None]:
    """Train, calibrate, and then predict-before-update on a held-out stream."""
    seed_everything(seed)
    Xtr, ytr = sample_train(
        split.X_train, split.y_train, train_size, seed, split.chronological
    )
    t0 = time.perf_counter()
    teachers = train_bpu_pool(
        bpu_kind, Xtr, ytr, epochs, seed + 100, split.chronological
    )
    student = train_classifier(
        make_model(spu_kind, Xtr.shape[1]),
        Xtr,
        ytr,
        epochs,
        seed + 700,
        reconstruction=False,
    )
    student = _train_student_with_matched_budget(
        student, teachers, Xtr, ytr, transfer, kd_epochs, seed + 900
    )
    _sync_device()
    initial_train_s = time.perf_counter() - t0
    models = teachers + [student]
    Pfit = _predict_matrix(models, split.X_cal_fit)
    cu = fit_cu(cu_name, Pfit, split.y_cal_fit)
    Pth = _predict_matrix(models, split.X_cal_thresh)
    threshold_active = float(tune_threshold(split.y_cal_thresh, predict_cu(cu, Pth)))
    initial_threshold = threshold_active

    # Bounded FIFO memory contains only labels that have been revealed from the
    # stream. The locked final-test labels are never used for updates/calibration.
    memory_x = np.empty((0, Xtr.shape[1]), dtype=np.float32)
    memory_y = np.empty((0,), dtype=np.int64)
    preds, labels, hard, window_rows = [], [], [], []
    refresh_interval = _effective_batch_interval(len(split.y_stream))
    next_refresh = refresh_interval
    prediction_seconds = update_seconds = refresh_seconds_total = 0.0
    n = len(split.y_stream)
    t_stream = time.perf_counter()
    train_feature_mean = (
        np.mean(Xtr, axis=0) if len(Xtr) else np.zeros(Xtr.shape[1], dtype=np.float32)
    )
    for st in range(0, n, STREAM_BATCH_SIZE):
        en = min(st + STREAM_BATCH_SIZE, n)
        xb, yb = split.X_stream[st:en], split.y_stream[st:en]
        # Test all records in the batch before revealing any labels in that batch.
        tp = time.perf_counter()
        p = predict_cu(cu, _predict_matrix(models, xb))
        _sync_device()
        prediction_seconds += time.perf_counter() - tp
        preds.extend(p.tolist())
        labels.extend(yb.tolist())
        hard.extend((p >= threshold_active).astype(int).tolist())
        memory_x = np.concatenate([memory_x, xb], axis=0)[-MEMORY_MAX:]
        memory_y = np.concatenate([memory_y, yb], axis=0)[-MEMORY_MAX:]
        if len(p):
            window_rows.append(
                {
                    "window_start": st,
                    "window_end": en,
                    "n": len(yb),
                    "positive_rate": float(np.mean(yb)),
                    "pr_auc": safe_pr_auc(yb, p),
                    "f1": float(f1_score(yb, p >= threshold_active, zero_division="0")),
                    "acc": float(accuracy_score(yb, p >= threshold_active)),
                    "threshold_active": float(threshold_active),
                    "feature_mean_abs_shift": float(
                        np.mean(np.abs(np.mean(xb, axis=0) - train_feature_mean))
                    ),
                }
            )
        # Online component updates only after this chunk has been scored.
        tu = time.perf_counter()
        online_update(models[-1], xb, yb)
        _sync_device()
        update_seconds += time.perf_counter() - tu

        if refresh_enabled and refresh_interval > 0 and en < n and en >= next_refresh:
            if len(memory_y) >= 64 and len(np.unique(memory_y)) == 2:
                rt0 = time.perf_counter()
                new_bpu = train_classifier(
                    make_model(bpu_kind, Xtr.shape[1]),
                    memory_x,
                    memory_y,
                    REFRESH_EPOCHS,
                    seed + 10000 + en,
                    reconstruction=True,
                )
                models = models[:-1] + [new_bpu] + [models[-1]]
                models = models[-(BPU_HISTORY_SIZE + 1) :]
                # Neural BPU->SPU converter: warm-start on revealed data then
                # distill the refreshed BPU pool into the active stream predictor.
                new_student = train_classifier(
                    make_model(spu_kind, Xtr.shape[1]),
                    memory_x,
                    memory_y,
                    REFRESH_EPOCHS,
                    seed + 11000 + en,
                    reconstruction=False,
                )
                new_student = _train_student_with_matched_budget(
                    new_student,
                    models[:-1],
                    memory_x,
                    memory_y,
                    transfer,
                    min(max(1, kd_epochs), 2),
                    seed + 12000 + en,
                )
                models[-1] = new_student
                _reset_online_optimizer(new_student)
                object.__setattr__(new_student, "_streamweave_online_labels", memory_y.copy())
                object.__setattr__(new_student, "_streamweave_pos_weight_online", _pos_weight(memory_y))
                cu = fit_cu(
                    cu_name, _predict_matrix(models, split.X_cal_fit), split.y_cal_fit
                )
                # Retune after each state change using the disjoint CAL_THRESH
                # partition, and use this threshold only for subsequent predictions.
                Pth = _predict_matrix(models, split.X_cal_thresh)
                threshold_active = float(
                    tune_threshold(split.y_cal_thresh, predict_cu(cu, Pth))
                )
                _sync_device()
                elapsed = time.perf_counter() - rt0
                refresh_seconds_total += elapsed
                window_rows[-1]["refresh_seconds"] = float(
                    window_rows[-1].get("refresh_seconds", 0.0) + elapsed
                )
                window_rows[-1]["threshold_after_refresh"] = threshold_active
            # Move the next trigger relative to the actual boundary. This prevents
            # multiple catch-up refreshes from training repeatedly on the same data.
            next_refresh = en + refresh_interval
    _sync_device()
    stream_seconds = time.perf_counter() - t_stream
    p_stream = np.asarray(preds, dtype=float)
    y_stream = np.asarray(labels, dtype=int)
    hard_stream = np.asarray(hard, dtype=int)
    # Probability-based metrics (log loss, ROC-AUC, PR-AUC) are computed from
    # the full prequential probability sequence. Accuracy/F1 below use each
    # event's threshold active at prediction time, including post-refresh retuning.
    preq: dict[str, Any] = metric_dict(y_stream, p_stream, initial_threshold)
    # Classification metrics use each event's threshold at its prediction time.
    if len(y_stream):
        preq["acc"] = float(accuracy_score(y_stream, hard_stream))
        preq["f1"] = float(f1_score(y_stream, hard_stream, zero_division="0"))
    preq.update(
        {
            "threshold_initial": float(initial_threshold),
            "threshold_final": float(threshold_active),
            "threshold_policy": "calibration_retuned_after_refresh"
            if refresh_enabled
            else "calibration_fixed",
            "stream_seconds": float(stream_seconds),
            "records_per_sec": float(len(y_stream) / max(stream_seconds, 1e-9)),
            "prediction_seconds": float(prediction_seconds),
            "update_seconds": float(update_seconds),
            "refresh_seconds_total": float(refresh_seconds_total),
            "bpu_refresh_count": int(
                sum(1 for r in window_rows if "refresh_seconds" in r)
            ),
            "threshold": float(
                initial_threshold
            ),  # compatibility alias for initial threshold
            "threshold_updates": int(
                sum(1 for r in window_rows if "threshold_after_refresh" in r)
            ),
        }
    )
    latency = (
        _measure_single_row_latency(
            lambda z: predict_cu(cu, _predict_matrix(models, z)), split.X_test
        )
        if run_test
        else {}
    )
    out = {
        "seed": seed,
        "bpu": bpu_kind,
        "spu": spu_kind,
        "transfer": transfer,
        "cu": cu_name,
        "train_seconds": float(initial_train_s),
        **{f"preq_{k}": v for k, v in preq.items()},
        **(
            {
                "preq_predict_latency_ms_mean": latency["predict_latency_ms_mean"],
                "preq_predict_latency_ms_p95": latency["predict_latency_ms_p95"],
            }
            if latency
            else {}
        ),
        "test_evaluated": bool(run_test),
        "success": True,
    }
    ptest = None
    if run_test:
        ttest = time.perf_counter()
        ptest = predict_cu(cu, _predict_matrix(models, split.X_test))
        _sync_device()
        out.update(
            {
                f"test_{k}": v
                for k, v in metric_dict(split.y_test, ptest, threshold_active).items()
            }
        )
        out["test_seconds"] = float(time.perf_counter() - ttest)
        out["n_test"] = len(split.y_test)
        out.update(latency)
    return out, window_rows, ptest


# ----------------------------- CSV/checkpoints ------------------------------
def _read_output_frame(path: Path) -> pd.DataFrame:
    """Read generated results without converting the literal transfer='None' to NA.

    pandas' default NA vocabulary includes the string "None". That value is a
    legitimate transfer-method name here, so result/checkpoint CSVs must disable
    the default NA vocabulary while still recognizing empty fields and NaN tokens.
    Source dataset readers intentionally keep their normal pandas NA behavior.
    """
    return pd.read_csv(
        path,
        keep_default_na=False,
        na_values=["", "nan", "NaN", "NAN"],
    )


def _all_str_equal(series: Any, value: object) -> bool:
    """True iff every stringified value in series equals str(value).

    Accepts a pd.Series, a single-column pd.DataFrame, or any scalar/iterable
    that pd.Series() can wrap. The union return of df[col] in pandas-stubs
    (Series | DataFrame) is normalised by _coerce_series before comparison,
    so call sites can pass df["col"] directly without narrowing.
    """
    s = _coerce_series(series)
    return bool(np.asarray(s.astype(str) == str(value), dtype=bool).all())


def _coerce_series(v: Any) -> pd.Series:
    """Return v as a pd.Series, collapsing single-column DataFrames as needed."""
    if isinstance(v, pd.Series):
        return v
    if isinstance(v, pd.DataFrame):
        if v.shape[1] != 1:
            raise ValueError(
                f"_coerce_series expected 1 column, got {v.shape[1]}"
            )
        return v.iloc[:, 0]
    return pd.Series(v)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return _read_output_frame(path).to_dict("records")
    except Exception as exc:
        raise RuntimeError(f"Cannot read checkpoint {path}: {exc}") from exc


def _write_rows(path: Path, rows: list[dict[str, Any]], identity: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if path.exists():
        old = _read_output_frame(path)
        out = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        out = new
    keys = [k for k in identity if k in out.columns]
    if keys:
        out = out.drop_duplicates(subset=keys, keep="last")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_current(
    path: Path, dataset: str, condition: str, run_sig: str, config_sig: str
) -> list[dict[str, Any]]:
    rows = _read_csv(path)
    out = []
    for r in rows:
        if (
            str(r.get("run_signature", "")) != run_sig
            or str(r.get("config_signature", "")) != config_sig
        ):
            continue
        if (
            str(r.get("dataset", "")) != dataset
            or str(r.get("condition", "")) != condition
        ):
            continue
        if (
            CURRENT_DATASET_FINGERPRINT
            and str(r.get("dataset_fingerprint", "")) != CURRENT_DATASET_FINGERPRINT
        ):
            continue
        out.append(r)
    return out


CURRENT_DATASET_FINGERPRINT = ""


def _save_row(path: Path, row: dict[str, Any], run_sig: str, config_sig: str) -> None:
    r = dict(row)
    r["run_signature"] = run_sig
    r["config_signature"] = config_sig
    if CURRENT_DATASET_FINGERPRINT and "dataset_fingerprint" not in r:
        r["dataset_fingerprint"] = CURRENT_DATASET_FINGERPRINT
    _write_rows(
        path,
        [r],
        [
            "run_signature",
            "config_signature",
            "dataset",
            "condition",
            "stage",
            "seed",
            "bpu",
            "spu",
            "transfer",
            "cu",
            "baseline",
            "window_start",
            "window_end",
        ],
    )


def _is_success(r: dict[str, Any]) -> bool:
    return str(r.get("success", "")).lower() in {"true", "1", "1.0"}


def _screen_missing_pairwise_coverage(
    rows: list[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Return factor-level pairs not represented by any successful screened triple."""
    required = set()
    for bpu in BPU_MODELS:
        for spu in SPU_MODELS:
            required.add(("bpu_spu", bpu, spu))
        for transfer in TRANSFERS:
            required.add(("bpu_transfer", bpu, transfer))
    for spu in SPU_MODELS:
        for transfer in TRANSFERS:
            required.add(("spu_transfer", spu, transfer))

    successful_triples = {
        (str(r.get("bpu", "")), str(r.get("spu", "")), str(r.get("transfer", "")))
        for r in rows
        if _is_success(r)
    }
    covered = set()
    for bpu, spu, transfer in successful_triples:
        covered.update(
            {
                ("bpu_spu", bpu, spu),
                ("bpu_transfer", bpu, transfer),
                ("spu_transfer", spu, transfer),
            }
        )
    return sorted(required - covered)


def _screen_cache(
    split: DataSplit, seed: int
) -> tuple[
    dict[str, list[TabularNet]],
    dict[str, TabularNet],
    dict[str, dict[str, np.ndarray]],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    Xs, ys = sample_train(
        split.X_train, split.y_train, SCREEN_TRAIN_SIZE, seed, split.chronological
    )
    teacher_pools = {}
    for i, kind in enumerate(BPU_MODELS):
        print(f"  [screen fit] BPU {kind}", flush=True)
        teacher_pools[kind] = train_bpu_pool(
            kind, Xs, ys, SCREEN_EPOCHS, seed + i * 10000, split.chronological
        )
    student_bases = {}
    for i, kind in enumerate(SPU_MODELS):
        print(f"  [screen fit] SPU base {kind}", flush=True)
        student_bases[kind] = train_classifier(
            make_model(kind, Xs.shape[1]),
            Xs,
            ys,
            SCREEN_EPOCHS,
            seed + 60000 + i * 1000,
            reconstruction=False,
        )
    xa, ya = subset_for_scoring(
        split.X_adapt, split.y_adapt, SCREEN_EVAL_MAX, seed + 44, split.chronological
    )
    part = {"cal_fit": split.X_cal_fit, "cal_thresh": split.X_cal_thresh, "adapt": xa}
    preds = {}
    for kind, pool in teacher_pools.items():
        preds[kind] = {name: _predict_matrix(pool, arr) for name, arr in part.items()}
    return teacher_pools, student_bases, preds, (xa, ya), (Xs, ys)


def _rank_screen_rows(
    current: list[dict[str, Any]],
    dataset: str,
    condition: str,
    run_sig: str,
    config_sig: str,
) -> None:
    good: pd.DataFrame = pd.DataFrame([r for r in current if _is_success(r)])
    if good.empty:
        raise RuntimeError(
            f"No successful budgeted-screen configurations for {dataset}/{condition}"
        )
    group = pd.DataFrame(
        good.groupby(["bpu", "spu", "transfer", "cu"], as_index=False).agg(
            adapt_pr_auc_mean=("adapt_pr_auc", "mean"),
            adapt_f1_mean=("adapt_f1", "mean"),
            adapt_logloss_mean=("adapt_logloss", "mean"),
            success_count=("success", "count"),
        )
    )
    group = group.sort_values(  # type: ignore
        by=["adapt_pr_auc_mean", "adapt_f1_mean", "adapt_logloss_mean"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    group.insert(0, "rank", np.arange(1, len(group) + 1))
    group.insert(0, "condition", condition)
    group.insert(0, "dataset", dataset)
    group["run_signature"] = run_sig
    group["config_signature"] = config_sig
    _write_rows(
        SAVE_DIR / "budgeted_screening_ranked.csv",
        group.to_dict("records"),
        ["run_signature", "config_signature", "dataset", "condition", "rank"],
    )


def _run_screen_dataset(
    dataset: str, condition: str, split: DataSplit, run_sig: str, config_sig: str
) -> list[dict[str, Any]]:
    path = SAVE_DIR / "budgeted_screening.csv"
    current = _load_current(path, dataset, condition, run_sig, config_sig)
    neural_combinations = _screen_neural_combinations()
    selected_neural = set(neural_combinations)
    expected = len(neural_combinations) * len(CUS)
    successes = [r for r in current if _is_success(r)]
    target_keys = {
        (bpu, spu, transfer, cu)
        for bpu, spu, transfer in neural_combinations
        for cu in CUS
    }
    recorded_keys = {
        (str(r.get("bpu")), str(r.get("spu")), str(r.get("transfer")), str(r.get("cu")))
        for r in current
    }
    prior_failures = [r for r in current if not _is_success(r)]
    prior_failure_fraction = len(prior_failures) / max(1, expected)
    prior_missing_pairs = _screen_missing_pairwise_coverage(current)
    if (
        target_keys.issubset(recorded_keys)
        and prior_failure_fraction <= SCREEN_MAX_FAILURE_FRACTION
        and not prior_missing_pairs
    ):
        print(
            f"[RESUME] budgeted screen loaded for {dataset}/{condition}; "
            f"{len(prior_failures)} logged failures within configured tolerance",
            flush=True,
        )
        _rank_screen_rows(current, dataset, condition, run_sig, config_sig)
        return current

    seed = SCREEN_SEED
    teachers, student_bases, teacher_preds, (xa, ya), (Xscreen, yscreen) = (
        _screen_cache(split, seed)
    )
    pred_adapt_size = len(ya)
    # Fit each student/transfer combination once; CU alternatives reuse the same
    # neural predictors and cached BPU predictions rather than retraining them.
    done = {
        (str(r.get("bpu")), str(r.get("spu")), str(r.get("transfer")), str(r.get("cu")))
        for r in successes
    }
    count = len(done)
    for bpu in BPU_MODELS:
        pool = teachers[bpu]
        base_pred = teacher_preds[bpu]
        for spu in SPU_MODELS:
            for transfer in TRANSFERS:
                if (bpu, spu, transfer) not in selected_neural:
                    continue
                try:
                    student = copy.deepcopy(student_bases[spu])
                    student = _train_student_with_matched_budget(
                        student,
                        pool,
                        Xscreen,
                        yscreen,
                        transfer,
                        SCREEN_KD_EPOCHS,
                        seed
                        + 71000
                        + BPU_MODELS.index(bpu) * 1000
                        + SPU_MODELS.index(spu) * 100
                        + TRANSFERS.index(transfer),
                    )
                    p_student = {
                        "cal_fit": predict_proba(student, split.X_cal_fit),
                        "cal_thresh": predict_proba(student, split.X_cal_thresh),
                        "adapt": predict_proba(student, xa),
                    }
                    for cu_name in CUS:
                        key = (bpu, spu, transfer, cu_name)
                        if tuple(map(str, key)) in done:
                            continue
                        try:
                            Pfit = np.column_stack(
                                [base_pred["cal_fit"], p_student["cal_fit"]]
                            )
                            Pth = np.column_stack(
                                [base_pred["cal_thresh"], p_student["cal_thresh"]]
                            )
                            Pad = np.column_stack(
                                [base_pred["adapt"], p_student["adapt"]]
                            )
                            combiner = fit_cu(cu_name, Pfit, split.y_cal_fit)
                            threshold = tune_threshold(
                                split.y_cal_thresh, predict_cu(combiner, Pth)
                            )
                            m = metric_dict(ya, predict_cu(combiner, Pad), threshold)
                            row = {
                                "dataset": dataset,
                                "condition": condition,
                                "stage": "screening",
                                "seed": seed,
                                "bpu": bpu,
                                "spu": spu,
                                "transfer": transfer,
                                "cu": cu_name,
                                "screen_train_n": len(Xscreen),
                                "adapt_n": pred_adapt_size,
                                "screen_epochs": SCREEN_EPOCHS,
                                "screen_kd_epochs": SCREEN_KD_EPOCHS,
                                "threshold": threshold,
                                **{f"adapt_{k}": v for k, v in m.items()},
                                "success": True,
                                "test_evaluated": False,
                            }
                        except Exception as exc:  # noqa: BLE001 — logged and re-surfaced or bounded per-record
                            row = {
                                "dataset": dataset,
                                "condition": condition,
                                "stage": "screening",
                                "seed": seed,
                                "bpu": bpu,
                                "spu": spu,
                                "transfer": transfer,
                                "cu": cu_name,
                                "success": False,
                                "test_evaluated": False,
                                "error": repr(exc),
                            }
                        _save_row(path, row, run_sig, config_sig)
                        done.add(key)
                        count = len(done)
                        if count % 10 == 0 or count == expected:
                            print(
                                f"[SCREEN] {dataset}/{condition}: {count}/{expected} configurations recorded",
                                flush=True,
                            )
                except Exception as exc:  # noqa: BLE001 — logged and re-surfaced or bounded per-record
                    # Record every CU for this failed neural combination explicitly.
                    for cu_name in CUS:
                        if (bpu, spu, transfer, cu_name) in done:
                            continue
                        row = {
                            "dataset": dataset,
                            "condition": condition,
                            "stage": "screening",
                            "seed": seed,
                            "bpu": bpu,
                            "spu": spu,
                            "transfer": transfer,
                            "cu": cu_name,
                            "success": False,
                            "test_evaluated": False,
                            "error": repr(exc),
                        }
                        _save_row(path, row, run_sig, config_sig)
                        done.add((bpu, spu, transfer, cu_name))
                        count = len(done)
                        if count % 10 == 0 or count == expected:
                            print(
                                f"[SCREEN] {dataset}/{condition}: {count}/{expected} configurations recorded",
                                flush=True,
                            )
    cleanup()
    current = _load_current(path, dataset, condition, run_sig, config_sig)
    _rank_screen_rows(current, dataset, condition, run_sig, config_sig)
    return current


def _top_candidates(rows: list[dict[str, Any]], k: int) -> list[dict[str, str]]:
    good: pd.DataFrame = pd.DataFrame([r for r in rows if _is_success(r)])
    if good.empty:
        raise RuntimeError("No successful screening rows")
    scores = pd.DataFrame(
        good.groupby(["bpu", "spu", "transfer", "cu"], as_index=False).agg(
            pr_auc=("adapt_pr_auc", "mean"),
            f1=("adapt_f1", "mean"),
            logloss=("adapt_logloss", "mean"),
        )
    )
    scores = scores.sort_values(  # type: ignore
        ["pr_auc", "f1", "logloss"], ascending=[False, False, True], kind="mergesort"
    )
    # First choose distinct neural triples so confirmation spends its expensive
    # multi-seed budget on architectural/transfer diversity, not multiple CUs
    # attached to a single lucky pilot model. Fill from the remaining ranked rows
    # only if there are fewer distinct neural triples than requested finalists.
    chosen = []
    seen_neural = set()
    for _, row in scores.iterrows():
        neural_key = (str(row["bpu"]), str(row["spu"]), str(row["transfer"]))
        if neural_key in seen_neural:
            continue
        seen_neural.add(neural_key)
        chosen.append({k2: str(row[k2]) for k2 in ["bpu", "spu", "transfer", "cu"]})
        if len(chosen) >= max(1, k):
            return chosen
    chosen_keys = {
        tuple(x[k2] for k2 in ["bpu", "spu", "transfer", "cu"]) for x in chosen
    }
    for _, row in scores.iterrows():
        candidate = {k2: str(row[k2]) for k2 in ["bpu", "spu", "transfer", "cu"]}
        if (
            tuple(candidate[k2] for k2 in ["bpu", "spu", "transfer", "cu"])
            in chosen_keys
        ):
            continue
        chosen.append(candidate)
        if len(chosen) >= max(1, k):
            break
    return chosen


def _run_confirmation(
    dataset: str,
    condition: str,
    split: DataSplit,
    finalists: list[dict[str, str]],
    run_sig: str,
    config_sig: str,
) -> list[dict[str, Any]]:
    path = SAVE_DIR / "confirmation.csv"
    old = _load_current(path, dataset, condition, run_sig, config_sig)
    for rank, cfg in enumerate(finalists, start=1):
        for seed in SEEDS:
            prior = next(
                (
                    r
                    for r in old
                    if str(r.get("seed")) == str(seed)
                    and all(
                        str(r.get(k)) == str(cfg[k])
                        for k in ["bpu", "spu", "transfer", "cu"]
                    )
                    and _is_success(r)
                ),
                None,
            )
            if prior:
                continue
            try:
                row, windows, _ = run_prequential(
                    split,
                    cfg["bpu"],
                    cfg["spu"],
                    cfg["transfer"],
                    cfg["cu"],
                    CONFIRM_TRAIN_SIZE,
                    CONFIRM_EPOCHS,
                    CONFIRM_KD_EPOCHS,
                    seed,
                    run_test=False,
                    refresh_enabled=True,
                )
                row.update(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "stage": "confirmation",
                        "candidate_rank": rank,
                        "chronological": split.chronological,
                        "task": split.task,
                        "test_evaluated": False,
                    }
                )
                for w in windows:
                    _save_row(
                        SAVE_DIR / "stream_windows.csv",
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "stage": "confirmation",
                            "candidate_rank": rank,
                            "seed": seed,
                            **cfg,
                            **w,
                        },
                        run_sig,
                        config_sig,
                    )
            except Exception as exc:  # noqa: BLE001 — logged and re-surfaced or bounded per-record
                row = {
                    "dataset": dataset,
                    "condition": condition,
                    "stage": "confirmation",
                    "seed": seed,
                    **cfg,
                    "candidate_rank": rank,
                    "success": False,
                    "test_evaluated": False,
                    "error": repr(exc),
                }
                traceback.print_exc()
            _save_row(path, row, run_sig, config_sig)
            old = _load_current(path, dataset, condition, run_sig, config_sig)
            print(
                f"[CONFIRM] {dataset}/{condition} seed={seed} cfg={cfg} success={row.get('success')} preq-PR-AUC={row.get('preq_pr_auc', float('nan'))}",
                flush=True,
            )
            cleanup()
    return _load_current(path, dataset, condition, run_sig, config_sig)


def _choose_confirmed(
    rows: list[dict[str, Any]], finalists: list[dict[str, str]]
) -> dict[str, str]:
    good = [r for r in rows if _is_success(r) and str(r.get("stage")) == "confirmation"]
    if not good:
        raise RuntimeError("No successful confirmation candidates")
    df = pd.DataFrame(good)
    grouped = pd.DataFrame(
        df.groupby(["bpu", "spu", "transfer", "cu"], as_index=False).agg(
            preq_pr_auc_mean=("preq_pr_auc", "mean"),
            preq_f1_mean=("preq_f1", "mean"),
            preq_f1_std=("preq_f1", "std"),
            preq_logloss_mean=("preq_logloss", "mean"),
            refresh_mean=("preq_bpu_refresh_count", "mean"),
            successful_seed_count=("seed", "nunique"),
        )
    )
    # Do not select a candidate from an incomplete set of confirmation seeds.
    grouped = grouped[grouped["successful_seed_count"] == len(SEEDS)]
    if grouped.empty:
        raise RuntimeError(
            "No finalist completed successfully for every confirmation seed"
        )
    grouped = grouped.sort_values(  # type: ignore
        ["preq_pr_auc_mean", "preq_f1_mean", "preq_logloss_mean"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    grouped.insert(0, "condition", str(rows[0].get("condition", "")))
    grouped.insert(0, "dataset", str(rows[0].get("dataset", "")))
    grouped["run_signature"] = str(rows[0].get("run_signature", ""))
    grouped["config_signature"] = str(rows[0].get("config_signature", ""))
    grouped["dataset_fingerprint"] = CURRENT_DATASET_FINGERPRINT
    _write_rows(
        SAVE_DIR / "confirmation_ranked.csv",
        grouped.to_dict("records"),
        [
            "run_signature",
            "config_signature",
            "dataset",
            "condition",
            "bpu",
            "spu",
            "transfer",
            "cu",
        ],
    )
    best = grouped.iloc[0]
    return {k: str(best[k]) for k in ["bpu", "spu", "transfer", "cu"]}


def _sync_device() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _measure_single_row_latency(
    predict_fn, X: np.ndarray, sample_cap: int = 64
) -> dict[str, float]:
    """Measure end-to-end one-row prediction latency (including host/device copies)."""
    X = np.asarray(X, dtype=np.float32)
    if len(X) == 0:
        return {
            "predict_latency_ms_mean": float("nan"),
            "predict_latency_ms_p95": float("nan"),
        }
    n = min(len(X), max(1, int(sample_cap)))
    # Warm up the exact callable before timing, without measuring warm-up effects.
    for i in range(min(3, n)):
        predict_fn(X[i : i + 1])
    _sync_device()
    times = []
    for i in range(n):
        _sync_device()
        t0 = time.perf_counter()
        predict_fn(X[i : i + 1])
        _sync_device()
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "predict_latency_ms_mean": float(np.mean(times)),
        "predict_latency_ms_p95": float(np.percentile(times, 95)),
    }


def _predict_in_batches(model: TabularNet, X: np.ndarray) -> np.ndarray:
    parts = [
        predict_proba(model, X[st : st + STREAM_BATCH_SIZE])
        for st in range(0, len(X), STREAM_BATCH_SIZE)
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def _run_batch_baseline(
    split: DataSplit, model_kind: str, seed: int
) -> tuple[dict[str, Any], np.ndarray]:
    """Static batch-only model: fit once, predict stream/test, never adapt online."""
    Xtr, ytr = sample_train(
        split.X_train, split.y_train, CONFIRM_TRAIN_SIZE, seed, split.chronological
    )
    t0 = time.perf_counter()
    model = train_classifier(
        make_model(model_kind, Xtr.shape[1]),
        Xtr,
        ytr,
        CONFIRM_EPOCHS,
        seed + 21000,
        reconstruction=True,
    )
    _sync_device()
    train_s = time.perf_counter() - t0
    threshold = tune_threshold(
        split.y_cal_thresh, predict_proba(model, split.X_cal_thresh)
    )
    t_stream = time.perf_counter()
    pstream = _predict_in_batches(model, split.X_stream)
    _sync_device()
    stream_s = time.perf_counter() - t_stream
    latency = _measure_single_row_latency(
        lambda z: predict_proba(model, z), split.X_test
    )
    ptest = _predict_in_batches(model, split.X_test)
    row = {
        "baseline": "BatchOnly",
        "model_kind": model_kind,
        "train_seconds": train_s,
        "stream_seconds": float(stream_s),
        "records_per_sec": float(len(split.y_stream) / max(stream_s, 1e-9)),
        "stream_prediction_seconds": float(stream_s),
        "stream_update_seconds": 0.0,
        "refresh_seconds_total": 0.0,
        "refresh_count": 0,
        **latency,
        **{
            f"preq_{k}": v
            for k, v in metric_dict(split.y_stream, pstream, threshold).items()
        },
        **{
            f"test_{k}": v
            for k, v in metric_dict(split.y_test, ptest, threshold).items()
        },
        "test_evaluated": True,
        "success": True,
    }
    return row, ptest


def _run_batch_refit_baseline(
    split: DataSplit, model_kind: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Batch-only adaptive baseline: predict first, then periodically refit from bounded recent data."""
    Xtr, ytr = sample_train(
        split.X_train, split.y_train, CONFIRM_TRAIN_SIZE, seed, split.chronological
    )
    t0 = time.perf_counter()
    model = train_classifier(
        make_model(model_kind, Xtr.shape[1]),
        Xtr,
        ytr,
        CONFIRM_EPOCHS,
        seed + 21500,
        reconstruction=True,
    )
    _sync_device()
    train_s = time.perf_counter() - t0
    threshold = tune_threshold(
        split.y_cal_thresh, predict_proba(model, split.X_cal_thresh)
    )
    initial_threshold = float(threshold)
    memory_x = np.asarray(Xtr[-min(MEMORY_MAX, len(Xtr)) :], dtype=np.float32).copy()
    memory_y = np.asarray(ytr[-min(MEMORY_MAX, len(ytr)) :], dtype=np.int64).copy()
    preds, labels, hard_preds, windows = [], [], [], []
    refresh_interval = _effective_batch_interval(len(split.y_stream))
    next_refresh = refresh_interval
    predict_s = 0.0
    refresh_s = 0.0
    refresh_count = 0
    train_feature_mean = (
        np.mean(Xtr, axis=0) if len(Xtr) else np.zeros(Xtr.shape[1], dtype=np.float32)
    )
    stream_t0 = time.perf_counter()
    for st in range(0, len(split.y_stream), STREAM_BATCH_SIZE):
        en = min(st + STREAM_BATCH_SIZE, len(split.y_stream))
        xb, yb = split.X_stream[st:en], split.y_stream[st:en]
        p0 = time.perf_counter()
        pp = predict_proba(model, xb)
        _sync_device()
        predict_s += time.perf_counter() - p0
        preds.extend(pp.tolist())
        labels.extend(yb.tolist())
        hard_preds.extend((pp >= threshold).astype(int).tolist())
        windows.append(
            {
                "window_start": st,
                "window_end": en,
                "n": len(yb),
                "positive_rate": float(np.mean(yb)),
                "pr_auc": safe_pr_auc(yb, pp),
                "f1": float(f1_score(yb, pp >= threshold, zero_division="0")),
                "acc": float(accuracy_score(yb, pp >= threshold)),
                "threshold_active": float(threshold),
                "feature_mean_abs_shift": float(
                    np.mean(np.abs(np.mean(xb, axis=0) - train_feature_mean))
                ),
            }
        )
        # Reveal the just-predicted labels, retain bounded data, and only then refit.
        memory_x = np.concatenate([memory_x, xb], axis=0)[-MEMORY_MAX:]
        memory_y = np.concatenate([memory_y, yb], axis=0)[-MEMORY_MAX:]
        if refresh_interval > 0 and en < len(split.y_stream) and en >= next_refresh:
            r0 = time.perf_counter()
            if len(memory_y) >= 64 and len(np.unique(memory_y)) == 2:
                model = train_classifier(
                    make_model(model_kind, Xtr.shape[1]),
                    memory_x,
                    memory_y,
                    REFRESH_EPOCHS,
                    seed + 23000 + en,
                    reconstruction=True,
                )
                threshold = tune_threshold(
                    split.y_cal_thresh, predict_proba(model, split.X_cal_thresh)
                )
                _sync_device()
                refresh_count += 1
                elapsed = time.perf_counter() - r0
                refresh_s += elapsed
                windows[-1]["refresh_seconds"] = elapsed
                windows[-1]["threshold_after_refresh"] = float(threshold)
            next_refresh = en + refresh_interval
    stream_s = time.perf_counter() - stream_t0
    pstream, ys = np.asarray(preds, dtype=float), np.asarray(labels, dtype=int)
    hard_stream = np.asarray(hard_preds, dtype=int)
    preq = metric_dict(ys, pstream, threshold)
    if len(ys):
        preq["acc"] = float(accuracy_score(ys, hard_stream))
        preq["f1"] = float(f1_score(ys, hard_stream, zero_division="0"))
    latency = _measure_single_row_latency(
        lambda z: predict_proba(model, z), split.X_test
    )
    ptest = _predict_in_batches(model, split.X_test)
    row = {
        "baseline": "BatchRefitOnly",
        "model_kind": model_kind,
        "train_seconds": train_s,
        "stream_seconds": float(stream_s),
        "records_per_sec": float(len(ys) / max(stream_s, 1e-9)),
        "stream_prediction_seconds": float(predict_s),
        "stream_update_seconds": 0.0,
        "refresh_seconds_total": float(refresh_s),
        "refresh_count": int(refresh_count),
        **latency,
        **{f"preq_{k}": v for k, v in preq.items()},
        "preq_threshold_initial": initial_threshold,
        "preq_threshold_final": float(threshold),
        "preq_threshold_policy": "calibration_retuned_after_batch_refit",
        **{
            f"test_{k}": v
            for k, v in metric_dict(split.y_test, ptest, threshold).items()
        },
        "test_evaluated": True,
        "success": True,
    }
    return row, windows, ptest


def _run_stream_only_baseline(
    split: DataSplit, model_kind: str, seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    """Online-only model: predict each batch before supervised online update; no BPU refresh."""
    Xtr, ytr = sample_train(
        split.X_train, split.y_train, CONFIRM_TRAIN_SIZE, seed, split.chronological
    )
    ttrain = time.perf_counter()
    model = train_classifier(
        make_model(model_kind, Xtr.shape[1]),
        Xtr,
        ytr,
        CONFIRM_EPOCHS,
        seed + 22000,
        reconstruction=False,
    )
    _sync_device()
    train_s = time.perf_counter() - ttrain
    threshold = tune_threshold(
        split.y_cal_thresh, predict_proba(model, split.X_cal_thresh)
    )
    preds, ys, windows = [], [], []
    predict_s = update_s = 0.0
    train_feature_mean = (
        np.mean(Xtr, axis=0) if len(Xtr) else np.zeros(Xtr.shape[1], dtype=np.float32)
    )
    stream_t0 = time.perf_counter()
    for st in range(0, len(split.y_stream), STREAM_BATCH_SIZE):
        en = min(st + STREAM_BATCH_SIZE, len(split.y_stream))
        xb, yb = split.X_stream[st:en], split.y_stream[st:en]
        t0 = time.perf_counter()
        p = predict_proba(model, xb)
        _sync_device()
        predict_s += time.perf_counter() - t0
        preds.extend(p.tolist())
        ys.extend(yb.tolist())
        windows.append(
            {
                "window_start": st,
                "window_end": en,
                "n": len(yb),
                "positive_rate": float(np.mean(yb)),
                "pr_auc": safe_pr_auc(yb, p),
                "f1": float(f1_score(yb, p >= threshold, zero_division="0")),
                "acc": float(accuracy_score(yb, p >= threshold)),
                "threshold_active": float(threshold),
                "feature_mean_abs_shift": float(
                    np.mean(np.abs(np.mean(xb, axis=0) - train_feature_mean))
                ),
            }
        )
        # This is the prequential label-reveal point: no current-batch label is
        # used until all predictions for that batch have been produced.
        u0 = time.perf_counter()
        online_update(model, xb, yb)
        _sync_device()
        update_s += time.perf_counter() - u0
    stream_s = time.perf_counter() - stream_t0
    p_stream, y_stream = np.asarray(preds), np.asarray(ys)
    latency = _measure_single_row_latency(
        lambda z: predict_proba(model, z), split.X_test
    )
    ptest = _predict_in_batches(model, split.X_test)
    row = {
        "baseline": "StreamOnly",
        "model_kind": model_kind,
        "train_seconds": float(train_s),
        "stream_seconds": float(stream_s),
        "records_per_sec": float(len(y_stream) / max(stream_s, 1e-9)),
        "stream_prediction_seconds": float(predict_s),
        "stream_update_seconds": float(update_s),
        "refresh_seconds_total": 0.0,
        "refresh_count": 0,
        **latency,
        **{
            f"preq_{k}": v
            for k, v in metric_dict(y_stream, p_stream, threshold).items()
        },
        **{
            f"test_{k}": v
            for k, v in metric_dict(split.y_test, ptest, threshold).items()
        },
        "test_evaluated": True,
        "success": True,
    }
    return row, windows, ptest


def _prediction_path(dataset: str, condition: str, method: str, seed: int) -> Path:
    return (
        SAVE_DIR
        / "test_predictions"
        / f"{dataset}__{condition}__{method}__seed{seed}.csv"
    )


def _save_test_predictions(
    dataset: str,
    condition: str,
    seed: int,
    method: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    run_sig: str,
    config_sig: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    out_path = _prediction_path(dataset, condition, method, seed)
    y_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    p_arr = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if len(y_arr) != len(p_arr) or not np.isfinite(p_arr).all():
        raise ValueError("Cannot save misaligned or non-finite test predictions")
    pred = (p_arr >= float(threshold)).astype(np.int64)
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "condition": condition,
            "seed": seed,
            "method": method,
            "test_index": np.arange(len(y_arr), dtype=np.int64),
            "y_true": y_arr,
            "probability": p_arr,
            "threshold": float(threshold),
            "predicted_label": pred,
            "correct": (pred == y_arr),
            "run_signature": run_sig,
            "config_signature": config_sig,
            "dataset_fingerprint": CURRENT_DATASET_FINGERPRINT,
        }
    )
    for key, value in (metadata or {}).items():
        frame[key] = value
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, out_path)


def _prediction_file_valid(
    path: Path,
    dataset: str,
    condition: str,
    method: str,
    seed: int,
    y_true: np.ndarray,
    run_sig: str,
    config_sig: str,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Validate every row before a prediction file is accepted for resume/paired tests."""
    if not path.exists():
        return False
    try:
        df = _read_output_frame(path)
    except Exception:  # noqa: BLE001 — file validation intentionally returns False
        return False
    required = {
        "dataset",
        "condition",
        "seed",
        "method",
        "test_index",
        "y_true",
        "probability",
        "run_signature",
        "config_signature",
        "dataset_fingerprint",
    }
    if not required.issubset(df.columns):
        return False
    y_arr = np.asarray(y_true, dtype=np.int64).reshape(-1)
    if len(df) != len(y_arr):
        return False

    if not _all_str_equal(df["dataset"], dataset):
            return False
    if not _all_str_equal(df["condition"], condition):
        return False
    if not bool(
        np.asarray(pd.to_numeric(df["seed"], errors="coerce") == int(seed)).all()
    ):
        return False
    if not _all_str_equal(df["method"], method):
        return False
    if not _all_str_equal(df["run_signature"], run_sig):
        return False
    if not _all_str_equal(df["config_signature"], config_sig):
        return False
    if not _all_str_equal(df["dataset_fingerprint"], CURRENT_DATASET_FINGERPRINT):
        return False
    try:
        test_idx = _coerce_series(pd.to_numeric(df["test_index"], errors="raise"))
        order = np.argsort(test_idx.to_numpy())
        df = df.iloc[order].reset_index(drop=True)
        idx = _coerce_series(
            pd.to_numeric(df["test_index"], errors="raise")
        ).to_numpy(dtype="int64")
        yy = _coerce_series(
            pd.to_numeric(df["y_true"], errors="raise")
        ).to_numpy(dtype="int64")
        pp = _coerce_series(
            pd.to_numeric(df["probability"], errors="raise")
        ).to_numpy(dtype="float64")
    except Exception:  # noqa: BLE001 — file validation intentionally returns False
        return False

    if not np.array_equal(idx, np.arange(len(y_arr), dtype=np.int64)):
        return False
    if not np.array_equal(yy, y_arr):
        return False
    if not np.isfinite(pp).all() or np.any((pp < 0) | (pp > 1)):
        return False
    for key, value in (metadata or {}).items():
        if key not in df.columns or not _all_str_equal(df[key], value):
            return False
    return True


def _write_paired_mcnemar(
    dataset: str,
    condition: str,
    temporal: bool,
    run_sig: str,
    config_sig: str,
    expected_cfg: dict[str, str] | None = None,
) -> None:
    """Paired exact McNemar comparisons on fixed test rows; temporal results flagged descriptive."""
    methods = ["StreamWeave", "BatchOnly", "BatchRefitOnly", "StreamOnly"]
    records = []
    for seed in SEEDS:
        loaded: dict[str, pd.DataFrame] = {}
        for method in methods:
            p = _prediction_path(dataset, condition, method, seed)
            if p.exists():
                df = _read_output_frame(p)
                required = {
                    "run_signature",
                    "config_signature",
                    "dataset_fingerprint",
                    "dataset",
                    "condition",
                    "seed",
                    "method",
                    "test_index",
                    "y_true",
                    "probability",
                    "correct",
                }
                if (
                    len(df)
                    and required.issubset(df.columns)
                    and _all_str_equal(df["run_signature"], run_sig)
                    and _all_str_equal(df["config_signature"], config_sig)
                    and _all_str_equal(
                        df["dataset_fingerprint"], CURRENT_DATASET_FINGERPRINT
                    )
                    and _all_str_equal(df["dataset"], dataset)
                    and _all_str_equal(df["condition"], condition)
                    and _all_str_equal(df["method"], method)
                    and bool(
                        np.asarray(
                            pd.to_numeric(df["seed"], errors="coerce") == seed
                        ).all()
                    )
                ):
                    expected_meta = (
                        (expected_cfg or {}) if method == "StreamWeave" else {}
                    )
                    if method == "StreamOnly" and expected_cfg:
                        expected_meta = {"model_kind": expected_cfg["spu"]}
                    elif method in {"BatchOnly", "BatchRefitOnly"} and expected_cfg:
                        expected_meta = {"model_kind": expected_cfg["bpu"]}
                    if any(
                        k not in df.columns or not _all_str_equal(df[k], v)
                        for k, v in expected_meta.items()
                    ):
                        continue
                    loaded[method] = df.sort_values("test_index").reset_index(drop=True)
        if "StreamWeave" not in loaded:
            continue
        for baseline in ("BatchOnly", "BatchRefitOnly", "StreamOnly"):
            if baseline not in loaded:
                continue
            a = loaded["StreamWeave"]
            b = loaded[baseline]
            if len(a) != len(b) or not np.array_equal(
                a["y_true"].to_numpy(), b["y_true"].to_numpy()
            ):
                raise RuntimeError(
                    f"Paired prediction alignment failure for {dataset}/{condition}/{seed}/{baseline}"
                )
            correct_a = a["correct"].astype(bool).to_numpy()
            correct_b = b["correct"].astype(bool).to_numpy()
            b_only = int(np.sum(correct_a & ~correct_b))
            c_only = int(np.sum(~correct_a & correct_b))
            total_discordant = b_only + c_only
            pvalue = (
                float(
                    binomtest(
                        b_only, total_discordant, 0.5, alternative="two-sided"
                    ).pvalue
                )
                if total_discordant
                else 1.0
            )
            records.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "seed": seed,
                    "comparison": f"StreamWeave_vs_{baseline}",
                    "n_test": len(a),
                    "hybrid_only_correct": b_only,
                    "baseline_only_correct": c_only,
                    "discordant_pairs": total_discordant,
                    "mcnemar_exact_p": pvalue,
                    "interpretation": "descriptive_only_temporal_dependence"
                    if temporal
                    else "paired_test_sample; repeated seeds reuse the same held-out records",
                    "run_signature": run_sig,
                    "config_signature": config_sig,
                    "dataset_fingerprint": CURRENT_DATASET_FINGERPRINT,
                }
            )
    if records:
        _write_rows(
            SAVE_DIR / "paired_mcnemar.csv",
            records,
            [
                "run_signature",
                "config_signature",
                "dataset",
                "condition",
                "seed",
                "comparison",
            ],
        )


def _write_tradeoff_tables(
    dataset: str, condition: str, run_sig: str, config_sig: str
) -> None:
    """Create paper-facing per-seed and aggregate accuracy/latency comparisons."""
    hybrid_rows = _load_current(
        SAVE_DIR / "final_test.csv", dataset, condition, run_sig, config_sig
    )
    baseline_rows = _load_current(
        SAVE_DIR / "baselines.csv", dataset, condition, run_sig, config_sig
    )
    records: list[dict[str, Any]] = []
    for r in hybrid_rows:
        if not (
            _is_success(r) and str(r.get("test_evaluated", "")).lower() in {"true", "1"}
        ):
            continue
        records.append(
            {
                "dataset": dataset,
                "condition": condition,
                "seed": int(float(r.get("seed", 0))),
                "method": "StreamWeave",
                "model_kind": f"BPU={r.get('bpu')};SPU={r.get('spu')}",
                "bpu": r.get("bpu"),
                "spu": r.get("spu"),
                "transfer": r.get("transfer"),
                "cu": r.get("cu"),
                **{
                    k: r.get(f"test_{k}")
                    for k in ("acc", "f1", "roc_auc", "pr_auc", "logloss", "threshold")
                },
                "train_seconds": r.get("train_seconds"),
                "stream_seconds": r.get("preq_stream_seconds"),
                "records_per_sec": r.get("preq_records_per_sec"),
                "predict_latency_ms_mean": r.get("preq_predict_latency_ms_mean"),
                "predict_latency_ms_p95": r.get("preq_predict_latency_ms_p95"),
                "stream_prediction_seconds": r.get("preq_prediction_seconds"),
                "stream_update_seconds": r.get("preq_update_seconds"),
                "refresh_seconds_total": r.get("preq_refresh_seconds_total"),
                "refresh_count": r.get("preq_bpu_refresh_count"),
            }
        )
    for r in baseline_rows:
        if not (
            _is_success(r) and str(r.get("test_evaluated", "")).lower() in {"true", "1"}
        ):
            continue
        method = str(r.get("baseline", "Baseline"))
        records.append(
            {
                "dataset": dataset,
                "condition": condition,
                "seed": int(float(r.get("seed", 0))),
                "method": method,
                "model_kind": r.get("model_kind"),
                "bpu": r.get("model_kind") if method != "StreamOnly" else "",
                "spu": r.get("model_kind") if method == "StreamOnly" else "",
                "transfer": "None",
                "cu": "None",
                **{
                    k: r.get(f"test_{k}")
                    for k in ("acc", "f1", "roc_auc", "pr_auc", "logloss", "threshold")
                },
                "train_seconds": r.get("train_seconds"),
                "stream_seconds": r.get("stream_seconds"),
                "records_per_sec": r.get("records_per_sec"),
                "predict_latency_ms_mean": r.get("predict_latency_ms_mean"),
                "predict_latency_ms_p95": r.get("predict_latency_ms_p95"),
                "stream_prediction_seconds": r.get("stream_prediction_seconds"),
                "stream_update_seconds": r.get("stream_update_seconds"),
                "refresh_seconds_total": r.get("refresh_seconds_total"),
                "refresh_count": r.get("refresh_count"),
            }
        )
    if not records:
        return
    by_seed = pd.DataFrame(records)
    by_seed = by_seed.drop_duplicates(
        subset=["dataset", "condition", "seed", "method"], keep="last"
    )
    by_seed["run_signature"] = run_sig
    by_seed["config_signature"] = config_sig
    _write_rows(
        SAVE_DIR / "tradeoff_by_seed.csv",
        by_seed.to_dict("records"),
        ["run_signature", "config_signature", "dataset", "condition", "seed", "method"],
    )
    numeric = [
        "acc",
        "f1",
        "roc_auc",
        "pr_auc",
        "logloss",
        "train_seconds",
        "stream_seconds",
        "records_per_sec",
        "predict_latency_ms_mean",
        "predict_latency_ms_p95",
        "stream_prediction_seconds",
        "stream_update_seconds",
        "refresh_seconds_total",
        "refresh_count",
    ]
    summaries = []
    for key, group in by_seed.groupby(
        ["dataset", "condition", "method"], dropna=False
    ):
        if not isinstance(key, tuple) or len(key) != 3:
            raise RuntimeError(f"Unexpected groupby key shape: {key!r}")
        ds, cond, method = key[0], key[1], key[2]
        group_frame: pd.DataFrame = (
            group if isinstance(group, pd.DataFrame) else pd.DataFrame(group)
        )
        row: dict[str, Any] = {
            "dataset": ds,
            "condition": cond,
            "method": method,
            "n_seeds": int(_coerce_series(group_frame["seed"]).nunique()),
            "run_signature": run_sig,
            "config_signature": config_sig,
        }
        for col in numeric:
            col_series = _coerce_series(
                pd.to_numeric(group_frame[col], errors="coerce")
            ).dropna()
            # Aggregate on a plain ndarray: Series.mean()/std() are declared as
            # Series | float in some pandas-stubs versions, and float(...) rejects
            # the union. Reducing an ndarray yields a numpy scalar that float()
            # accepts without ambiguity, and semantics are identical.
            arr = col_series.to_numpy(dtype="float64")
            row[f"{col}_mean"] = float(arr.mean()) if arr.size else float("nan")
            row[f"{col}_std"] = (
                float(arr.std(ddof=1))
                if arr.size > 1
                else (0.0 if arr.size == 1 else float("nan"))
            )
        summaries.append(row)
    _write_rows(
        SAVE_DIR / "tradeoff_summary.csv",
        summaries,
        ["run_signature", "config_signature", "dataset", "condition", "method"],
    )


def _final_evaluate(
    dataset: str,
    condition: str,
    split: DataSplit,
    cfg: dict[str, str],
    run_sig: str,
    config_sig: str,
) -> list[dict[str, Any]]:
    path = SAVE_DIR / "final_test.csv"
    baselines_path = SAVE_DIR / "baselines.csv"
    old = _load_current(path, dataset, condition, run_sig, config_sig)
    base_old = _load_current(baselines_path, dataset, condition, run_sig, config_sig)
    windows_path = SAVE_DIR / "stream_windows.csv"
    for seed in SEEDS:
        pred_file = _prediction_path(dataset, condition, "StreamWeave", seed)
        predictions_present = _prediction_file_valid(
            pred_file,
            dataset,
            condition,
            "StreamWeave",
            seed,
            split.y_test,
            run_sig,
            config_sig,
            {
                "bpu": cfg["bpu"],
                "spu": cfg["spu"],
                "transfer": cfg["transfer"],
                "cu": cfg["cu"],
            },
        )
        prior = next(
            (
                r
                for r in old
                if str(r.get("seed")) == str(seed)
                and all(
                    str(r.get(k)) == str(cfg[k])
                    for k in ["bpu", "spu", "transfer", "cu"]
                )
                and _is_success(r)
                and str(r.get("test_evaluated", "")).lower() in {"true", "1"}
            ),
            None,
        )
        if not prior or not predictions_present:
            try:
                row, windows, ptest = run_prequential(
                    split,
                    cfg["bpu"],
                    cfg["spu"],
                    cfg["transfer"],
                    cfg["cu"],
                    CONFIRM_TRAIN_SIZE,
                    CONFIRM_EPOCHS,
                    CONFIRM_KD_EPOCHS,
                    seed,
                    run_test=True,
                    refresh_enabled=True,
                )
                row["seed"] = seed
                row.update(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "stage": "final_test",
                        "chronological": split.chronological,
                        "task": split.task,
                        "selected_config": json.dumps(cfg, sort_keys=True),
                    }
                )
                if ptest is None:
                    raise RuntimeError(
                        "run_prequential returned no test predictions despite run_test=True"
                    )
                _save_test_predictions(
                    dataset,
                    condition,
                    seed,
                    "StreamWeave",
                    split.y_test,
                    ptest,
                    row["test_threshold"],
                    run_sig,
                    config_sig,
                    {
                        "bpu": cfg["bpu"],
                        "spu": cfg["spu"],
                        "transfer": cfg["transfer"],
                        "cu": cfg["cu"],
                    },
                )
                for w in windows:
                    _save_row(
                        windows_path,
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "stage": "final_test",
                            "seed": seed,
                            **cfg,
                            **w,
                        },
                        run_sig,
                        config_sig,
                    )
            except Exception as exc:  # noqa: BLE001 — per-seed failure is logged and recorded
                row = {
                    "dataset": dataset,
                    "condition": condition,
                    "stage": "final_test",
                    "seed": seed,
                    **cfg,
                    "success": False,
                    "test_evaluated": False,
                    "error": repr(exc),
                }
                traceback.print_exc()
            _save_row(path, row, run_sig, config_sig)
            old = _load_current(path, dataset, condition, run_sig, config_sig)
        # The comparisons are prespecified; evaluate each baseline on the same
        # fixed test partition, but never use these numbers for configuration selection.
        for bname in ("BatchOnly", "BatchRefitOnly", "StreamOnly"):
            bpred_path = _prediction_path(dataset, condition, bname, seed)
            expected_kind = cfg["spu"] if bname == "StreamOnly" else cfg["bpu"]
            expected_metadata = {"model_kind": expected_kind}
            bpred_present = _prediction_file_valid(
                bpred_path,
                dataset,
                condition,
                bname,
                seed,
                split.y_test,
                run_sig,
                config_sig,
                expected_metadata,
            )
            exists = next(
                (
                    r
                    for r in base_old
                    if str(r.get("seed")) == str(seed)
                    and str(r.get("baseline")) == bname
                    and str(r.get("model_kind")) == expected_kind
                    and _is_success(r)
                ),
                None,
            )
            if exists and bpred_present:
                continue
            try:
                if bname == "BatchOnly":
                    br, bp = _run_batch_baseline(split, cfg["bpu"], seed)
                    bw = []
                elif bname == "BatchRefitOnly":
                    br, bw, bp = _run_batch_refit_baseline(split, cfg["bpu"], seed)
                else:
                    br, bw, bp = _run_stream_only_baseline(split, cfg["spu"], seed)
                for w in bw:
                    _save_row(
                        windows_path,
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "stage": f"baseline_{bname.lower()}",
                            "seed": seed,
                            "bpu": cfg["bpu"] if bname != "StreamOnly" else "",
                            "spu": cfg["spu"] if bname == "StreamOnly" else "",
                            "transfer": "None",
                            "cu": "None",
                            **w,
                        },
                        run_sig,
                        config_sig,
                    )
                _save_test_predictions(
                    dataset,
                    condition,
                    seed,
                    bname,
                    split.y_test,
                    bp,
                    br["test_threshold"],
                    run_sig,
                    config_sig,
                    {"model_kind": expected_kind},
                )
                br.update(
                    {
                        "dataset": dataset,
                        "condition": condition,
                        "seed": seed,
                        "stage": "baseline",
                        "chronological": split.chronological,
                        "task": split.task,
                    }
                )
            except Exception as exc:  # noqa: BLE001 — per-baseline failure is logged and recorded
                br = {
                    "dataset": dataset,
                    "condition": condition,
                    "seed": seed,
                    "baseline": bname,
                    "model_kind": expected_kind,
                    "success": False,
                    "test_evaluated": False,
                    "error": repr(exc),
                }
                traceback.print_exc()
            _save_row(baselines_path, br, run_sig, config_sig)
            base_old = _load_current(
                baselines_path, dataset, condition, run_sig, config_sig
            )
        cleanup()
    _write_paired_mcnemar(
        dataset, condition, split.chronological, run_sig, config_sig, cfg
    )
    _write_tradeoff_tables(dataset, condition, run_sig, config_sig)
    # A resumed run may have records from an earlier winner if it was interrupted
    # after selection. Return only the configuration selected in this run so the
    # summary cannot silently mix different winning configurations.
    current = _load_current(path, dataset, condition, run_sig, config_sig)
    return [
        r
        for r in current
        if all(str(r.get(k)) == str(cfg[k]) for k in ["bpu", "spu", "transfer", "cu"])
    ]


def _report_drift(
    split: DataSplit, dataset: str, condition: str, run_sig: str, config_sig: str
) -> None:
    # These are descriptive natural time-window rates; the script never injects
    # or simulates artificial drift. Temporal evaluation does not, by itself,
    # prove a concept-drift event; the resulting profiles quantify observed change.
    partitions = [
        ("train", split.y_train),
        ("cal_fit", split.y_cal_fit),
        ("cal_thresh", split.y_cal_thresh),
        ("adapt", split.y_adapt),
        ("stream", split.y_stream),
        ("test", split.y_test),
    ]
    rows = []
    for name, y in partitions:
        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "partition": name,
                "n": len(y),
                "positive_rate": float(np.mean(y)),
                "chronological": split.chronological,
                "task": split.task,
            }
        )
    _write_rows(
        SAVE_DIR / "partition_profile.csv",
        [{**r, "run_signature": run_sig, "config_signature": config_sig} for r in rows],
        ["run_signature", "config_signature", "dataset", "condition", "partition"],
    )


def run_smoke_tests() -> None:
    """Fast local smoke tests without downloading data or launching the screen."""
    seed_everything(77)
    screen_design = _screen_neural_combinations()
    all_pair_tags = set()
    selected_pair_tags = set()
    for bpu in BPU_MODELS:
        for spu in SPU_MODELS:
            all_pair_tags.add(("bpu_spu", bpu, spu))
    for bpu in BPU_MODELS:
        for transfer in TRANSFERS:
            all_pair_tags.add(("bpu_transfer", bpu, transfer))
    for spu in SPU_MODELS:
        for transfer in TRANSFERS:
            all_pair_tags.add(("spu_transfer", spu, transfer))
    for bpu, spu, transfer in screen_design:
        selected_pair_tags.update(
            {
                ("bpu_spu", bpu, spu),
                ("bpu_transfer", bpu, transfer),
                ("spu_transfer", spu, transfer),
            }
        )
    assert all_pair_tags == selected_pair_tags, (
        "screening design lost pairwise factor coverage"
    )
    assert len(screen_design) == len(set(screen_design)), (
        "screening design has duplicate triples"
    )
    rng = np.random.default_rng(77)
    simulated_screen_rows = [
        {"bpu": bpu, "spu": spu, "transfer": transfer, "cu": cu, "success": True}
        for bpu, spu, transfer in screen_design
        for cu in CUS
    ]
    assert not _screen_missing_pairwise_coverage(simulated_screen_rows), (
        "complete successful screen unexpectedly lacks pairwise coverage"
    )
    simulated_screen_rows[0]["success"] = False
    assert not _screen_missing_pairwise_coverage(simulated_screen_rows), (
        "an isolated CU failure incorrectly removed successful neural-triple coverage"
    )
    n, d = 800, 12
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = ((X[:, 0] + 0.5 * X[:, 1] + rng.normal(size=n) * 0.5) > 0.25).astype(np.int64)
    # Check split disjointness and preprocessing-shaped arrays.
    raw = pd.DataFrame(X, columns=[f"x{i}" for i in range(d)])
    split = preprocess_split(raw, y, False, "smoke_test", 17)
    # pandas StringDtype must be treated as categorical and one-hot encoded.
    string_raw = pd.DataFrame(
        {
            "numeric": rng.normal(size=n).astype(np.float32),
            "kind": pd.Series(
                np.where(np.arange(n) % 2, "alpha", "beta"), dtype="string"
            ),
        }
    )
    string_split = preprocess_split(string_raw, y, False, "string_dtype_smoke", 18)
    assert any(name.startswith("kind_") for name in string_split.feature_names), (
        "pandas StringDtype column was not encoded as categorical"
    )
    assert np.isfinite(string_split.X_train).all(), (
        "StringDtype preprocessing produced non-finite values"
    )
    # Dataset semantic manifest must preserve schema and target counts atomically.
    import tempfile

    old_save_for_fingerprint_test = globals()["SAVE_DIR"]
    try:
        with tempfile.TemporaryDirectory(prefix="streamweave-fingerprint-smoke-") as td:
            globals()["SAVE_DIR"] = Path(td)
            _write_dataset_fingerprint_manifest(
                "Smoke", raw, y, split, "smoke_task", "smoke-run", "smoke-config"
            )
            fingerprint_manifest = json.loads(
                (Path(td) / "DATASET_FINGERPRINTS.json").read_text(encoding="utf-8")
            )
            assert (
                fingerprint_manifest["datasets"]["Smoke"]["semantic_sha256"]
                == split.dataset_fingerprint
            )
            assert (
                fingerprint_manifest["datasets"]["Smoke"]["target_counts"]["0"]
                + fingerprint_manifest["datasets"]["Smoke"]["target_counts"]["1"]
                == n
            )
    finally:
        globals()["SAVE_DIR"] = old_save_for_fingerprint_test
    lengths = [
        len(split.y_train),
        len(split.y_cal_fit),
        len(split.y_cal_thresh),
        len(split.y_adapt),
        len(split.y_stream),
        len(split.y_test),
    ]
    assert (
        sum(lengths) == n
        and len(np.unique(np.concatenate([split.y_train, split.y_cal_fit]))) == 2
    )
    probs: np.ndarray = np.empty(0, dtype=np.float32)
    for model_kind in SPU_MODELS:
        model = train_classifier(
            make_model(model_kind, d),
            split.X_train[:256],
            split.y_train[:256],
            1,
            78 + SPU_MODELS.index(model_kind),
            reconstruction=False,
        )
        probs = predict_proba(model, split.X_cal_fit)
        assert len(probs) == len(split.y_cal_fit) and np.isfinite(probs).all(), (
            f"bad predictions from {model_kind}"
        )
    P = np.column_stack([probs, probs])
    for cu in CUS:
        params = fit_cu(cu, P, split.y_cal_fit)
        pp = predict_cu(params, P)
        assert np.isfinite(pp).all() and len(pp) == len(split.y_cal_fit)
    # Test every distillation method with tiny networks/data.
    teachers = [
        train_classifier(
            make_model("SmallMLP", d),
            split.X_train[:256],
            split.y_train[:256],
            1,
            80 + i,
            reconstruction=False,
        )
        for i in range(2)
    ]
    for tr in TRANSFERS:
        base = train_classifier(
            make_model("SmallMLP", d),
            split.X_train[:256],
            split.y_train[:256],
            1,
            90,
            reconstruction=False,
        )
        dist = _kd_epoch(
            base, teachers, split.X_train[:128], split.y_train[:128], tr, 1, 91
        )
        if tr in {"FitNets", "ContrastiveRepKD"}:
            assert dist.training, f"student left in eval mode for {tr}"
        p = predict_proba(dist, split.X_cal_thresh[:32])
        assert np.isfinite(p).all()
    # The no-KD control must perform its matched extra supervised epoch.
    control = train_classifier(
        make_model("SmallMLP", d),
        split.X_train[:128],
        split.y_train[:128],
        1,
        111,
        reconstruction=False,
    )
    before = [q.detach().cpu().clone() for q in control.parameters()]
    control = _train_student_with_matched_budget(
        control, teachers, split.X_train[:128], split.y_train[:128], "None", 1, 112
    )
    assert any(
        not torch.equal(a, b.detach().cpu())
        for a, b in zip(before, control.parameters())
    ), "matched training budget did not update the no-KD control"
    # Regression checks for findings that previously could silently corrupt labels.
    mapped = _nbaiot_explicit_labels(
        pd.Series([0, 1, 0, 1]), Path("numeric-label-test.csv")
    )
    assert mapped.tolist() == [0, 1, 0, 1]
    try:
        safe_pr_auc(np.asarray([0, 1]), np.asarray([0.2]))
    except ValueError:
        pass
    else:
        raise AssertionError("safe_pr_auc did not reject length mismatch")
    old_interval = BATCH_INTERVAL
    try:
        # With 300 records and 256-row batches, a refresh can affect the second batch.
        globals()["BATCH_INTERVAL"] = 8192
        assert _effective_batch_interval(300) == 100
        assert _effective_batch_interval(256) == 0
    finally:
        globals()["BATCH_INTERVAL"] = old_interval
    # Literal transfer method "None" must survive CSV write/read and resume.
    import tempfile

    old_save_for_csv_test = globals()["SAVE_DIR"]
    try:
        with tempfile.TemporaryDirectory(prefix="streamweave-csv-none-smoke-") as td:
            globals()["SAVE_DIR"] = Path(td)
            checkpoint = Path(td) / "none_transfer_checkpoint.csv"
            _write_rows(
                checkpoint,
                [
                    {
                        "dataset": "Smoke",
                        "condition": "iid",
                        "transfer": "None",
                        "success": True,
                    }
                ],
                ["dataset", "condition", "transfer"],
            )
            _write_rows(
                checkpoint,
                [
                    {
                        "dataset": "Smoke",
                        "condition": "iid",
                        "transfer": "StandardKD",
                        "success": True,
                    }
                ],
                ["dataset", "condition", "transfer"],
            )
            saved = _read_csv(checkpoint)
            assert {str(row["transfer"]) for row in saved} == {"None", "StandardKD"}, (
                "CSV round-trip converted transfer='None' into a missing value"
            )
    finally:
        globals()["SAVE_DIR"] = old_save_for_csv_test

    # Existing but malformed local OpenML files must fail closed, not trigger a network fallback.

    original_candidates = globals()["_local_candidates"]
    original_record_input = globals()["_record_input_file"]
    try:
        with tempfile.TemporaryDirectory(prefix="streamweave-local-parse-smoke-") as td:
            bad_path = Path(td) / "adult.csv"
            bad_path.write_text("one_column\nvalue\n", encoding="utf-8")
            globals()["_local_candidates"] = lambda _name: [bad_path]
            globals()["_record_input_file"] = lambda path, source_url=None: (
                "smoke-sha256"
            )
            try:
                _load_local_openml_dataset("Adult")
            except RuntimeError as exc:
                assert "none was usable" in str(exc) and str(bad_path) in str(exc)
            else:
                raise AssertionError(
                    "Unusable local file silently fell back as if absent"
                )
    finally:
        globals()["_local_candidates"] = original_candidates
        globals()["_record_input_file"] = original_record_input

    # Exercise headered and headerless N-BaIoT loaders in an isolated temporary directory.
    old_data_dir = globals()["DATA_DIR"]
    old_quota = os.environ.get("STREAMWEAVE_NBAIOT_ROWS")
    old_chunk_size = os.environ.get("STREAMWEAVE_NBAIOT_CHUNK_SIZE")
    try:
        with tempfile.TemporaryDirectory(prefix="streamweave-nbaiot-smoke-") as td:
            temp_root = Path(td)
            nbroot = temp_root / "N-BaIoT"
            nbroot.mkdir()
            globals()["DATA_DIR"] = temp_root
            os.environ["STREAMWEAVE_NBAIOT_ROWS"] = "1000"
            os.environ["STREAMWEAVE_NBAIOT_CHUNK_SIZE"] = "7"
            nrng = np.random.default_rng(101)
            a = nrng.normal(size=(12, 115))
            b = nrng.normal(size=(13, 115))
            pd.DataFrame(a).to_csv(
                nbroot / "benign_traffic.csv", index=False, header=False
            )
            pd.DataFrame(b).to_csv(nbroot / "mirai.csv", index=False, header=False)
            nx, ny = _read_nbaiot()
            assert nx.shape == (25, 115) and set(np.unique(ny)) == {0, 1}
            for f in nbroot.glob("*.csv"):
                f.unlink()
            (temp_root / "source_sha256_manifest.json").unlink(missing_ok=True)
            cols = [f"feature_{i}" for i in range(115)]
            pd.DataFrame(a, columns=cols).to_csv(
                nbroot / "benign_traffic.csv", index=False
            )
            pd.DataFrame(b, columns=cols).to_csv(nbroot / "mirai.csv", index=False)
            nx, ny = _read_nbaiot()
            assert nx.shape == (25, 115) and set(np.unique(ny)) == {0, 1}

            # Force reservoir sampling and exercise the headerless 115+label schema.
            for f in nbroot.glob("*.csv"):
                f.unlink()
            (temp_root / "source_sha256_manifest.json").unlink(missing_ok=True)
            os.environ["STREAMWEAVE_NBAIOT_ROWS"] = "10"
            os.environ["STREAMWEAVE_NBAIOT_CHUNK_SIZE"] = "5"
            pd.DataFrame(np.column_stack([a, np.zeros(len(a))])).to_csv(
                nbroot / "benign_traffic.csv", index=False, header=False
            )
            pd.DataFrame(np.column_stack([b, np.ones(len(b))])).to_csv(
                nbroot / "mirai.csv", index=False, header=False
            )
            nx, ny = _read_nbaiot()
            assert nx.shape == (10, 115) and set(np.unique(ny)) == {0, 1}
    finally:
        globals()["DATA_DIR"] = old_data_dir
        if old_quota is None:
            os.environ.pop("STREAMWEAVE_NBAIOT_ROWS", None)
        else:
            os.environ["STREAMWEAVE_NBAIOT_ROWS"] = old_quota
        if old_chunk_size is None:
            os.environ.pop("STREAMWEAVE_NBAIOT_CHUNK_SIZE", None)
        else:
            os.environ["STREAMWEAVE_NBAIOT_CHUNK_SIZE"] = old_chunk_size

    # Filename-derived labels must fail closed when both classes appear in path.
    try:
        _infer_nbaiot_filename_label(Path("benign_attack_mirai.csv"))
    except ValueError:
        pass
    else:
        raise AssertionError("ambiguous N-BaIoT filename was not rejected")

    # Verify that resumable test predictions require matching signatures,
    # dataset fingerprints, and baseline-model metadata.
    old_save_dir = globals()["SAVE_DIR"]
    old_fingerprint = globals()["CURRENT_DATASET_FINGERPRINT"]
    try:
        with tempfile.TemporaryDirectory(prefix="streamweave-prediction-smoke-") as td:
            globals()["SAVE_DIR"] = Path(td)
            globals()["CURRENT_DATASET_FINGERPRINT"] = "smoke-dataset-fingerprint"
            yy = np.asarray([0, 1, 0, 1], dtype=np.int64)
            pp = np.asarray([0.1, 0.9, 0.2, 0.8], dtype=np.float32)
            _save_test_predictions(
                "Smoke",
                "iid",
                0,
                "BatchOnly",
                yy,
                pp,
                0.5,
                "smoke-run",
                "smoke-config",
                {"model_kind": "MLP"},
            )
            pred_path = _prediction_path("Smoke", "iid", "BatchOnly", 0)
            assert _prediction_file_valid(
                pred_path,
                "Smoke",
                "iid",
                "BatchOnly",
                0,
                yy,
                "smoke-run",
                "smoke-config",
                {"model_kind": "MLP"},
            )
            assert not _prediction_file_valid(
                pred_path,
                "Smoke",
                "iid",
                "BatchOnly",
                0,
                yy,
                "stale-run",
                "smoke-config",
                {"model_kind": "MLP"},
            )
            assert not _prediction_file_valid(
                pred_path,
                "Smoke",
                "iid",
                "BatchOnly",
                0,
                yy,
                "smoke-run",
                "smoke-config",
                {"model_kind": "GRU"},
            )
    finally:
        globals()["SAVE_DIR"] = old_save_dir
        globals()["CURRENT_DATASET_FINGERPRINT"] = old_fingerprint
    print("SMOKE TESTS PASSED", flush=True)


def _write_dataset_fingerprint_manifest(
    dataset: str,
    X: pd.DataFrame,
    y: np.ndarray,
    split: DataSplit,
    task: str,
    run_sig: str,
    config_sig: str,
) -> None:
    """Persist content-level dataset identity, including OpenML-loaded data."""
    path = SAVE_DIR / "DATASET_FINGERPRINTS.json"
    try:
        prior = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:  # noqa: BLE001 — corrupted manifest is treated as absent and rewritten
        prior = {}
    if (
        prior.get("run_signature") != run_sig
        or prior.get("config_signature") != config_sig
        or not isinstance(prior.get("datasets"), dict)
    ):
        prior = {
            "run_signature": run_sig,
            "config_signature": config_sig,
            "fingerprint_policy": (
                "SHA-256 over a schema/task/chronology descriptor, pandas value hashes "
                "preserving row order, and mapped binary target bytes; this pins the "
                "loaded semantic dataset content but is not a byte hash of a remote artifact."
            ),
            "datasets": {},
        }

    known_openml_ids = {
        "CreditCardFraud": 1597,
        "Adult": 1590,
        "BankMarketing": 1461,
        "Electricity": 151,
    }
    if dataset in known_openml_ids:
        source = {
            "kind": "OpenML dataset ID, unless a recognized local file was found first",
            "data_id": known_openml_ids[dataset],
        }
    elif dataset == "Covertype":
        source = {"kind": "scikit-learn fetch_covtype cache"}
    elif dataset == "N-BaIoT":
        source = {
            "kind": "local N-BaIoT CSV files; byte hashes in source_sha256_manifest.json"
        }
    elif dataset in {"Skin", "NSL-KDD"}:
        source = {
            "kind": "local dataset files; byte hashes in source_sha256_manifest.json when available"
        }
    else:
        source = {"kind": "source resolved by load_dataset"}
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2)
    prior["datasets"][dataset] = {
        "source": source,
        "semantic_sha256": split.dataset_fingerprint,
        "rows": len(y),
        "raw_feature_count": int(X.shape[1]),
        "raw_columns": [str(c) for c in X.columns],
        "target_counts": {"0": int(counts[0]), "1": int(counts[1])},
        "task": str(task),
        "chronological": bool(split.chronological),
        "preprocessed_feature_count": len(split.feature_names),
    }
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(prior, indent=2), encoding="utf-8")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StreamWeave staged neural batch-stream experiment"
    )
    parser.add_argument(
        "--smoke-test", action="store_true", help="run fast local code-path tests only"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="write the run plan and exit"
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="fail immediately unless PyTorch can see a CUDA GPU",
    )
    args = parser.parse_args()
    if args.require_cuda and not torch.cuda.is_available():
        parser.error(
            "--require-cuda was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
        )
    if args.smoke_test:
        run_smoke_tests()
        return
    ensure_dirs()
    run_sig = hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()
    config = {
        "datasets": DATASETS_TO_RUN,
        "bpu_models": BPU_MODELS,
        "spu_models": SPU_MODELS,
        "transfers": TRANSFERS,
        "cus": CUS,
        "screening_strategy": "deterministic_greedy_pairwise_cover_over_BPU_SPU_transfer_then_seeded_fill; all_CUs_evaluated_for_each_selected_neural_triple; top_K_confirmed_on_multiple_seeds",
        "screen_neural_budget_requested": SCREEN_NEURAL_BUDGET,
        "screen_neural_budget_effective": len(_screen_neural_combinations()),
        "screen_max_failure_fraction": SCREEN_MAX_FAILURE_FRACTION,
        "screen_neural_combinations": [list(x) for x in _screen_neural_combinations()],
        "screen_train_size": SCREEN_TRAIN_SIZE,
        "confirm_train_size": CONFIRM_TRAIN_SIZE,
        "screen_epochs": SCREEN_EPOCHS,
        "screen_kd_epochs": SCREEN_KD_EPOCHS,
        "confirm_epochs": CONFIRM_EPOCHS,
        "confirm_kd_epochs": CONFIRM_KD_EPOCHS,
        "refresh_epochs": REFRESH_EPOCHS,
        "online_epochs": ONLINE_EPOCHS,
        "seeds": SEEDS,
        "top_k": SCREEN_TOP_K,
        "screen_eval_max": SCREEN_EVAL_MAX,
        "fractions": FRACTIONS,
        "history_size": BPU_HISTORY_SIZE,
        "batch_interval": BATCH_INTERVAL,
        "memory_max": MEMORY_MAX,
        "stream_batch_size": STREAM_BATCH_SIZE,
        "base_lr": BASE_LR,
        "online_lr": ONLINE_LR,
        "weight_decay": WEIGHT_DECAY,
        "pos_weight_cap": POS_WEIGHT_CAP,
        "class_weight_policy": f"offline pos_weight capped at {POS_WEIGHT_CAP:g}; online pos_weight recomputed from rolling revealed-label memory",
        "num_workers": NUM_WORKERS,
        "master_seed": MASTER_SEED,
        "screen_seed": SCREEN_SEED,
        "device": str(DEVICE),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "python": sys.version,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "drift_policy": "no synthetic drift injection; chronological sequence preserved only for datasets with defensible source order",
        "baseline_modes": ["BatchOnly", "BatchRefitOnly", "StreamOnly", "StreamWeave"],
        "baseline_selection_policy": "BatchOnly and BatchRefitOnly use the selected BPU architecture; StreamOnly uses the selected SPU architecture. These are controlled-capacity comparisons, not independently hyperparameter-optimized baselines. BatchRefitOnly initializes its bounded refit memory from the tail of the offline training partition, so it assumes that training history is retained and is not a cold-start stream baseline.",
        "runtime_metrics": [
            "train_seconds",
            "stream_seconds",
            "records_per_sec",
            "predict_latency_ms_mean",
            "predict_latency_ms_p95",
            "stream_update_seconds",
            "refresh_seconds_total",
        ],
        "spu_architectures": SPU_MODELS,
        "sequence_model_scope": "GRU and TCN operate over ordered features within a row; they are feature-sequence architectures, not recurrent models over prior stream events",
        "latency_protocol": "predict_latency_ms_mean/p95 (and preq-prefixed equivalents) are single-row end-to-end host-call latency including host/device transfer, kernel-launch, and synchronization overhead; they are not model-only GPU inference time. They are measured on up to 64 rows after warm-up. records_per_sec is measured separately over the actual batch-processing stream loop.",
        "natural_drift_protocol": "no synthetic perturbation; source chronology retained only for Electricity and credit-card transaction Time when available; Bank Marketing is not treated as temporal because its common source lacks a reliable full timestamp; per-window label prevalence and feature-mean shift are descriptive signals, not proof of concept drift",
        "refresh_interval_policy": "configured maximum interval capped at roughly one-third of stream size; refresh only at chunk boundaries before the final chunk and at most once per chunk",
    }
    config_sig = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    plan = {
        **config,
        "run_signature": run_sig,
        "config_signature": config_sig,
        "full_factorial_configurations_per_dataset": len(BPU_MODELS)
        * len(SPU_MODELS)
        * len(TRANSFERS)
        * len(CUS),
        "screening_configurations_per_dataset": len(_screen_neural_combinations())
        * len(CUS),
        "screening_neural_training_combinations_per_dataset": len(
            _screen_neural_combinations()
        ),
        "screening_pairwise_coverage": "all levels of BPU-SPU, BPU-transfer, and SPU-transfer pairs are represented; higher-order interactions are not exhaustively evaluated; if any screen row fails, successful neural triples must still cover every factor pair",
        "screen_failure_policy": "allow at most screen_max_failure_fraction of individual rows to fail only if all factor-level pairwise combinations remain represented among successful neural triples; failures are logged and excluded; set STREAMWEAVE_SCREEN_MAX_FAILURE_FRACTION=0 for strict all-rows-must-pass behavior",
        "openml_fingerprint_policy": "OpenML source artifacts are fetched by explicit data_id; the loaded raw feature/label content, schema, task, and chronology flag are represented by the per-dataset SHA-256 semantic fingerprint written into results and summaries. This is not a byte-for-byte hash of the remote ARFF payload.",
        "training_budget_fairness": "transfer=None receives supervised BCE continuation for the same number of additional epochs and the same 256-row maximum batch size as KD methods, at screening, confirmation initialization, and refresh; objectives and wall-clock compute can still differ by method",
        "resource_profile": {
            "preset": "laptop_friendly_budgeted_screen",
            "screen_train_rows_max": SCREEN_TRAIN_SIZE,
            "screen_epochs": SCREEN_EPOCHS,
            "screen_neural_triples": len(_screen_neural_combinations()),
            "screening_rows_per_dataset": len(_screen_neural_combinations()) * len(CUS),
            "full_grid_rows_per_dataset": len(BPU_MODELS)
            * len(SPU_MODELS)
            * len(TRANSFERS)
            * len(CUS),
            "screening_row_reduction_fraction": round(
                1.0
                - (len(_screen_neural_combinations()) * len(CUS))
                / (len(BPU_MODELS) * len(SPU_MODELS) * len(TRANSFERS) * len(CUS)),
                4,
            ),
            "confirmation_train_rows_max": CONFIRM_TRAIN_SIZE,
            "confirmation_epochs_before_matched_transfer_budget": CONFIRM_EPOCHS,
            "confirmation_seeds": SEEDS,
        },
        "baseline_modes": ["BatchOnly", "BatchRefitOnly", "StreamOnly", "StreamWeave"],
        "screening_primary_metric": "adapt_pr_auc",
        "confirmation_primary_metric": "preq_pr_auc",
        "selection_uses_test": False,
        "device": str(DEVICE),
        "cuda_device": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "paper_reference": "Pishgoo et al., A hybrid distributed batch-stream processing approach for anomaly detection, Information Sciences 543 (2021), 309-327, DOI 10.1016/j.ins.2020.07.026",
        "nbaiot_source_note": "N-BaIoT is part of the base paper's benchmark set and must be placed in data/N-BaIoT; its loader samples at most STREAMWEAVE_NBAIOT_ROWS rows (default 855932).",
        "nbaiot_mixed_schema_policy": "Headered and headerless files are normalized to semantic names and feature_N names respectively; incompatible mixed schemas fail closed on feature-name/order mismatch instead of unioning or truncating columns.",
        "input_hash_cache_policy": "SHA-256 is reused only when resolved path, size_bytes, and mtime_ns match the source manifest; remove the manifest entry to force rehashing. Same-size edits that preserve mtime may evade cache invalidation.",
        "local_downloads": {
            "Adult": ["phpMawTba", "phpMawTba.arff", "adult.arff"],
            "BankMarketing": ["phpkIxskf", "phpkIxskf.arff", "bank-marketing.arff"],
            "Electricity": ["electricity-normalized.arff", "electricity.arff"],
        },
        "method_note": "HDT_Analog is an equal-weight neural probability ensemble analogue, not the original decision-tree converter/HDT; KD is used to initialize/refit the SPU on BPU refresh.",
    }
    (SAVE_DIR / "EXPERIMENT_PLAN.json").write_text(
        json.dumps(plan, indent=2), encoding="utf-8"
    )
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    print(
        f"StreamWeave staged-screening run | device={DEVICE} | datasets={DATASETS_TO_RUN}",
        flush=True,
    )
    print(
        f"Screen configs per dataset/condition: {plan['screening_configurations_per_dataset']} "
        f"(full grid would be {plan['full_factorial_configurations_per_dataset']}); results={SAVE_DIR}",
        flush=True,
    )
    summaries = []
    failures = []
    for dataset in DATASETS_TO_RUN:
        print(f"\n=== Loading {dataset} ===", flush=True)
        try:
            X, y, temporal, task = load_dataset(dataset)
            if len(np.unique(y)) != 2:
                raise ValueError("Target mapping did not yield both binary classes")
            split = preprocess_split(X, y, temporal, task, MASTER_SEED)
            global CURRENT_DATASET_FINGERPRINT
            CURRENT_DATASET_FINGERPRINT = split.dataset_fingerprint
            _write_dataset_fingerprint_manifest(
                dataset, X, y, split, task, run_sig, config_sig
            )
            condition = "natural_temporal" if temporal else "stratified_iid"
            _report_drift(split, dataset, condition, run_sig, config_sig)
            print(
                f"{dataset}: n={len(y)}, d={split.X_train.shape[1]}, positive_rate={np.mean(y):.6f}, condition={condition}, task={task}",
                flush=True,
            )
            screen_rows = _run_screen_dataset(
                dataset, condition, split, run_sig, config_sig
            )
            good_n = sum(1 for r in screen_rows if _is_success(r))
            expected = len(_screen_neural_combinations()) * len(CUS)
            print(
                f"Budgeted-screen coverage: {good_n}/{expected} successful configurations "
                f"({len(_screen_neural_combinations())} neural triples, all {len(CUS)} CUs per triple)",
                flush=True,
            )
            failed_screen_rows = [r for r in screen_rows if not _is_success(r)]
            screen_failure_fraction = len(failed_screen_rows) / max(1, expected)
            missing_pairs = _screen_missing_pairwise_coverage(screen_rows)
            if (
                screen_failure_fraction > SCREEN_MAX_FAILURE_FRACTION
                or missing_pairs
                or good_n + len(failed_screen_rows) < expected
            ):
                failed_keys = [
                    {k: r.get(k) for k in ("bpu", "spu", "transfer", "cu", "error")}
                    for r in failed_screen_rows
                ]
                raise RuntimeError(
                    f"Budgeted screening failed acceptance policy: successful={good_n}/{expected}, "
                    f"failure_fraction={screen_failure_fraction:.4f}, "
                    f"allowed={SCREEN_MAX_FAILURE_FRACTION:.4f}, missing_pairwise_combinations={missing_pairs[:10]}. "
                    f"Inspect budgeted_screening.csv. Failed rows (first 10): {failed_keys[:10]}"
                )
            if failed_screen_rows:
                print(
                    f"[SCREEN WARNING] proceeding with {len(failed_screen_rows)} logged failures "
                    f"({screen_failure_fraction:.2%}) because the failure budget and pairwise coverage checks pass",
                    flush=True,
                )
                failure_log = SAVE_DIR / "screening_config_failures.csv"
                _write_rows(
                    failure_log,
                    [
                        {
                            **r,
                            "run_signature": run_sig,
                            "config_signature": config_sig,
                            "dataset_fingerprint": CURRENT_DATASET_FINGERPRINT,
                        }
                        for r in failed_screen_rows
                    ],
                    [
                        "run_signature",
                        "config_signature",
                        "dataset",
                        "condition",
                        "stage",
                        "bpu",
                        "spu",
                        "transfer",
                        "cu",
                    ],
                )
            finalists = _top_candidates(screen_rows, SCREEN_TOP_K)
            print(
                f"Top-K candidates from budgeted development-only screen: {finalists}",
                flush=True,
            )
            conf_rows = _run_confirmation(
                dataset, condition, split, finalists, run_sig, config_sig
            )
            successful_cfgs = {
                tuple(str(r.get(k)) for k in ["bpu", "spu", "transfer", "cu"])
                for r in conf_rows
                if _is_success(r)
            }
            if not successful_cfgs:
                raise RuntimeError("All confirmation candidates failed")
            winner = _choose_confirmed(conf_rows, finalists)
            print(
                f"Selected configuration from prequential development stream: {winner}",
                flush=True,
            )
            final_rows = _final_evaluate(
                dataset, condition, split, winner, run_sig, config_sig
            )
            final_ok = [
                r
                for r in final_rows
                if _is_success(r)
                and str(r.get("test_evaluated", "")).lower() in {"true", "1"}
            ]
            if len(final_ok) != len(SEEDS):
                raise RuntimeError(
                    f"Final-test evaluation is incomplete for {dataset}: "
                    f"{len(final_ok)}/{len(SEEDS)} selected-configuration seeds succeeded. "
                    "No aggregate test metric will be reported; rerun to resume failed seeds."
                )
            baseline_rows = _load_current(
                SAVE_DIR / "baselines.csv", dataset, condition, run_sig, config_sig
            )
            missing_baseline_seeds = []
            for baseline_name in ("BatchOnly", "BatchRefitOnly", "StreamOnly"):
                expected_kind = (
                    winner["spu"] if baseline_name == "StreamOnly" else winner["bpu"]
                )
                for seed in SEEDS:
                    baseline_ok = any(
                        _is_success(r)
                        and str(r.get("seed")) == str(seed)
                        and str(r.get("baseline")) == baseline_name
                        and str(r.get("model_kind")) == expected_kind
                        and str(r.get("test_evaluated", "")).lower() in {"true", "1"}
                        for r in baseline_rows
                    )
                    if not baseline_ok:
                        missing_baseline_seeds.append((baseline_name, seed))
            if missing_baseline_seeds:
                raise RuntimeError(
                    f"Baseline final-test evaluation is incomplete for {dataset}: "
                    f"{missing_baseline_seeds}. No complete comparative summary will be reported; "
                    "rerun to resume failed baseline runs."
                )
            summary = {
                "dataset": dataset,
                "condition": condition,
                "task": task,
                "chronological": temporal,
                "n": len(y),
                "features_after_preprocess": len(split.feature_names),
                "source_fingerprint": split.dataset_fingerprint,
                "screening_successful": good_n,
                "screening_planned": expected,
                "screening_failed": len(failed_screen_rows),
                "screening_failure_fraction": screen_failure_fraction,
                "screening_missing_pairwise_combinations": missing_pairs,
                "full_factorial_configurations_not_evaluated": plan[
                    "full_factorial_configurations_per_dataset"
                ]
                - expected,
                "final_selected": winner,
                "final_test_runs_successful": len(final_ok),
                "final_test_seed_count_expected": len(SEEDS),
                "all_prespecified_baseline_seed_runs_successful": True,
                "final_test_f1_mean": float(np.mean([r["test_f1"] for r in final_ok]))
                if final_ok
                else None,
                "final_test_f1_std": float(
                    np.std([r["test_f1"] for r in final_ok], ddof=1)
                )
                if len(final_ok) > 1
                else None,
                "final_test_pr_auc_mean": float(
                    np.nanmean([r["test_pr_auc"] for r in final_ok])
                )
                if final_ok
                else None,
                "test_partition_never_used_for_selection": True,
            }
            summaries.append(summary)
            (SAVE_DIR / f"{dataset}_{condition}_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001 — per-dataset failure is recorded and the loop continues
            failure = {
                "dataset": dataset,
                "success": False,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            failures.append(failure)
            _write_rows(
                SAVE_DIR / "experiment_failures.csv",
                [{**failure, "run_signature": run_sig, "config_signature": config_sig}],
                ["run_signature", "config_signature", "dataset"],
            )
            print(f"FAILED {dataset}: {exc}", flush=True)
            traceback.print_exc()
        finally:
            cleanup()
    report = {
        **plan,
        "completed": summaries,
        "failures": failures,
        "scientific_caveat": "Chronological order is used as observed; natural drift is measured descriptively and is not fabricated or guaranteed. Repeated seeds on one fixed holdout are not independent datasets and should not be used as independent statistical units.",
    }
    (SAVE_DIR / "FINAL_MANIFEST.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"\nFinished. Results: {SAVE_DIR}", flush=True)


if __name__ == "__main__":
    main()
