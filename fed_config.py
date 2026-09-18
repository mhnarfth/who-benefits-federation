"""
fed_config.py -- Single source of truth for the INDIS 2026 short paper:
"Learning Without Centralizing: Federated Representation Learning for
 Multi-Network Traffic Telemetry"

DESIGN PRINCIPLE (learned the hard way earlier in this project):
    Derived values, never duplicated constants. Any quantity that can be
    computed from another quantity IS computed, so two constants can never
    silently disagree and produce results that look plausible but are wrong.

Everything downstream (Notebooks 01, 02, 03) imports from here. No notebook
redefines a hyperparameter inline.
"""

from pathlib import Path
import json
import math
import os
import random

import numpy as np

# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path("/lustre/work/ramamurthy/mhnarfth/indis2026-federated")

# ---- SOURCE DATA -----------------------------------------------------------
# CESNET3: confirmed path from prior work in this project.
#   NOTE: this lives on /mnt/nrdstor while the project lives on /lustre/work.
#   Different filesystems -- verify_paths() checks it is actually reachable
#   from this compute node.
CESNET_ROOT = Path("/mnt/nrdstor/ramamurthy/mhnarfth/Datasets/cesnet")
CESNET_INSTITUTIONS_DIR = CESNET_ROOT / "institutions" / "agg_1_hour"
CESNET_TIMES_PATH = CESNET_ROOT / "times" / "times_1_hour.csv"

# >>> ACTION REQUIRED <<<
# Internet2: all prior notebooks used the RELATIVE path "outputs/ch4/".
# Set the absolute path here. Expected contents:
#   hourly_<router>_processed_with_anomalies.parquet   (10 files)
INTERNET2_DIR = Path("/mnt/nrdstor/ramamurthy/mhnarfth/Datasets/internet2_old_processed/hourly_parquet")

# >>> ACTION REQUIRED <<<
# ISP ISP: format is UNKNOWN to this config. Notebook 01 Cell 1 runs a
# diagnostic against whatever is at this path and reports the real schema.
# It may be a single file (.parquet/.csv) or a directory of files.
ISP_PATH = Path("/mnt/nrdstor/ramamurthy/mhnarfth/Datasets/isp/processed/isp_hourly.parquet")

# After running the Cell 1 diagnostic, set these to the true column names.
# # Leave as None to let the loader attempt auto-detection.
# ISP_TIME_COL = None      # e.g. "t_first", "timestamp", "time"
# ISP_VALUE_COL = None     # e.g. "in_packets", "n_packets", "packets"
# ISP_IS_PREAGGREGATED = None  # True if already hourly; False if raw flows

# --- Explicit column mapping for ISP ---
ISP_TIME_COL = "start_datetime"
ISP_VALUE_COL = "Packets"
ISP_IS_PREAGGREGATED = True

# ---- OUTPUTS ---------------------------------------------------------------
OUT_ROOT = PROJECT_ROOT / "outputs" / "indis2026"
CLIENTS_DIR = OUT_ROOT / "clients"
MODELS_DIR = OUT_ROOT / "models"
METRICS_DIR = OUT_ROOT / "metrics"
CHARTS_DIR = OUT_ROOT / "charts"

ALL_OUTPUT_DIRS = [OUT_ROOT, CLIENTS_DIR, MODELS_DIR, METRICS_DIR, CHARTS_DIR]


def ensure_dirs():
    """Create all output directories. Idempotent."""
    for d in ALL_OUTPUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def verify_paths(verbose=True):
    """
    Check every required path before any work begins.

    Returns a dict of {label: (path, exists, is_required)}. Notebook 01 Cell 0
    calls this and raises if any REQUIRED path is missing. This exists because
    silently proceeding on a missing path produces plausible-looking garbage.
    """
    checks = {
        "PROJECT_ROOT":        (PROJECT_ROOT, True),
        "INTERNET2_DIR":       (INTERNET2_DIR, True),
        "CESNET_INSTITUTIONS": (CESNET_INSTITUTIONS_DIR, True),
        "CESNET_TIMES":        (CESNET_TIMES_PATH, True),
        "ISP_PATH":           (ISP_PATH, False),  # optional: holdout probe only
    }
    results = {}
    for label, (path, required) in checks.items():
        exists = path.exists()
        results[label] = {"path": str(path), "exists": exists, "required": required}

    if verbose:
        print("=" * 72)
        print("PATH VERIFICATION")
        print("=" * 72)
        for label, r in results.items():
            mark = "OK  " if r["exists"] else ("FAIL" if r["required"] else "WARN")
            req = "required" if r["required"] else "optional"
            print(f"  [{mark}] {label:22s} ({req:8s}) {r['path']}")
        print("=" * 72)

    missing_required = [k for k, r in results.items() if r["required"] and not r["exists"]]
    if missing_required:
        raise FileNotFoundError(
            f"Required path(s) not found: {missing_required}. "
            f"Edit the path constants at the top of fed_config.py."
        )
    return results


# =============================================================================
# 2. CLIENT REGISTRY
# =============================================================================

# Internet2: all 10 routers.
#   NOTE: unlike the concept-drift study (which excluded Boston and Reno for
#   insufficient POST-CUTOFF data), this task has no post-cutoff window
#   requirement, so all 10 routers qualify as federated clients. See
#   min_hours_required() below for the derived eligibility rule.
INTERNET2_ROUTERS = [
    "atlanta", "batonrouge", "boston", "dallas", "elpaso",
    "jackson", "jacksonville", "louisville", "phoenix", "reno",
]

# CESNET3: the 10 highest-volume institutions, ordered by mean hourly
# n_packets. Established and reused from prior work in this project.
CESNET_PRIMARY_IDS = ["2", "0", "49", "1", "48", "3", "61", "53", "57", "50"]

# For the client-scalability study (K = 20 -> 40 -> 60 -> 100). CESNET has 283
# institutions available; Notebook 01 can extend the registry by re-ranking
# the full directory. Set to None to use only CESNET_PRIMARY_IDS.
CESNET_SCALABILITY_TOP_N = None  # e.g. 50 to pull the top-50 institutions

# Temporal trim applied to every CESNET series (9 weeks), for scope
# comparability with the 57-day Internet2 window.
CESNET_TRIM_HOURS = 1512

# ISP is a HOLDOUT PROBE, never a training client. See min_hours_required():
# at L=24 its 49-hour series yields 0 validation and 0 test windows, so it
# cannot participate in training. It is embedded through frozen encoders at
# evaluation time as an unseen-network transfer test.
ISP_ROLE = "holdout_probe"
TRAIN_CLIENT_ROLE = "train_client"

NETWORK_CLASSES = ["Internet2", "CESNET3", "ISP"]


# =============================================================================
# 3. WINDOWING AND SPLITS
# =============================================================================

# Generate all three window lengths in one pass (costs kilobytes) so the
# Notebook 02 window-length ablation needs no return trip to data prep.
WINDOW_LENGTHS = [12, 24, 48]
PRIMARY_WINDOW_LENGTH = 24   # one full diurnal cycle
WINDOW_STRIDE = 1

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Minimum windows required in EVERY split for a client to be eligible.
MIN_WINDOWS_PER_SPLIT = 30

# Apply log1p before z-scoring. Traffic is heavy-tailed (established in the
# CCDF analysis of prior work); this stops the autoencoder from spending
# capacity reconstructing outlier spikes.
LOG_TRANSFORM = True


def min_hours_required(window_length, min_windows=None):
    """
    DERIVED client-eligibility threshold -- never hardcode this.

    For a split of ratio r on a series of T hours, the split has floor(r*T)
    hours, yielding (floor(r*T) - L + 1) windows at stride 1. Requiring at
    least `min_windows` in the SMALLEST split gives:

        T >= ceil( (min_windows + L - 1) / min(r_val, r_test) )

    At L=24, min_windows=30, r=0.15  ->  T >= 354 hours.
    """
    if min_windows is None:
        min_windows = MIN_WINDOWS_PER_SPLIT
    smallest_ratio = min(VAL_RATIO, TEST_RATIO)
    return int(math.ceil((min_windows + window_length - 1) / smallest_ratio))


def n_windows(n_hours, window_length, stride=None):
    """Number of stride-`stride` windows extractable from `n_hours` points."""
    if stride is None:
        stride = WINDOW_STRIDE
    if n_hours < window_length:
        return 0
    return (n_hours - window_length) // stride + 1


def split_sizes(total_hours):
    """Chronological 70/15/15 split point sizes for a series of `total_hours`."""
    n_train = int(TRAIN_RATIO * total_hours)
    n_val = int((TRAIN_RATIO + VAL_RATIO) * total_hours) - n_train
    n_test = total_hours - n_train - n_val
    return n_train, n_val, n_test


# =============================================================================
# 4. MODEL HYPERPARAMETERS
# =============================================================================

# Symmetric MLP autoencoder: L -> 128 -> 64 -> d -> 64 -> 128 -> L
#
# SIZING RATIONALE (state this in the paper): the binding constraint is
# DATA per client (~1,000 windows), not GPU capacity. At ~24K parameters this
# is roughly 24 params/sample -- near the sensible ceiling before local
# overfitting destabilizes FedAvg weight averaging. The latent-dim ablation
# demonstrates this empirically rather than by assertion.
ENCODER_HIDDEN_DIMS = [128, 64]
LATENT_DIM = 8

# Ablation grids (Notebook 02 sweeps these; the H200 makes them ~free)
LATENT_DIM_ABLATION = [4, 8, 16, 32]
WINDOW_LENGTH_ABLATION = WINDOW_LENGTHS
LOCAL_EPOCHS_ABLATION = [1, 5, 10]

ACTIVATION = "relu"
OUTPUT_ACTIVATION = None  # linear -- inputs are z-scored, can be negative


# =============================================================================
# 5. FEDERATED / TRAINING HYPERPARAMETERS
# =============================================================================

FED_ROUNDS = 100
LOCAL_EPOCHS = 5
CLIENT_PARTICIPATION_FRACTION = 1.0  # full participation; K=20 needs no sampling

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
OPTIMIZER = "adam"

# DERIVED epoch budgets -- this is what guarantees the three regimes are
# compared at equal total optimization budget. Hardcoding these separately is
# exactly the class of bug that bit this project before.
TOTAL_EPOCH_BUDGET = FED_ROUNDS * LOCAL_EPOCHS   # 500
CENTRALIZED_MAX_EPOCHS = TOTAL_EPOCH_BUDGET
LOCAL_ONLY_MAX_EPOCHS = TOTAL_EPOCH_BUDGET

# Early stopping applies to the CENTRALIZED and LOCAL-ONLY baselines (standard
# validation-based patience). Standard FedAvg does NOT early-stop locally;
# instead the global model is selected by best mean validation loss across
# rounds. This asymmetry is deliberate and should be stated in the paper.
EARLY_STOP_PATIENCE = 25

REGIMES = ["Centralized", "Federated", "Local-only"]


# =============================================================================
# 6. SEEDS AND DETERMINISM
# =============================================================================

SEEDS = [42, 43, 44, 45, 46]   # 5 seeds -> every headline number gets mean +/- std
PRIMARY_SEED = 42


def set_seed(seed=None):
    """Seed python, numpy, and torch (incl. CUDA) for reproducibility."""
    if seed is None:
        seed = PRIMARY_SEED
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


# =============================================================================
# 7. PLOTTING
# =============================================================================

NETWORK_COLORS = {
    "Internet2": "#4C72B0",   # blue
    "CESNET3":   "#DD8452",   # orange
    "ISP":      "#55A868",   # green
}

REGIME_COLORS = {
    "Centralized": "#2F4B7C",
    "Federated":   "#D45087",
    "Local-only":  "#7F7F7F",
}

FIGURE_DPI = 300


def setup_matplotlib():
    """
    Apply project-wide plot styling.

    ASCII-ONLY RULE: matplotlib's PDF backend silently drops unicode en/em
    dashes from the embedded text layer, which breaks text search and indexing
    in the published PDF. This project has hit that bug twice. Use plain
    ASCII hyphens in every title, label, and annotation.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = FIGURE_DPI
    plt.rcParams["font.size"] = 11
    plt.rcParams["savefig.bbox"] = "tight"


# def save_figure(fig_name, formats=("pdf", "png")):
#     """Save the current matplotlib figure to CHARTS_DIR in all formats."""
#     import matplotlib.pyplot as plt
#     ensure_dirs()
#     paths = []
#     for fmt in formats:
#         p = CHARTS_DIR / f"{fig_name}.{fmt}"
#         plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
#         paths.append(p)
#     plt.close()
#     for p in paths:
#         print(f"  saved: {p.name}")
#     return paths

def save_figure(fig_name, formats=("pdf", "png"), show=True):
    """Save the current matplotlib figure to CHARTS_DIR in all formats and optionally display it."""
    import matplotlib.pyplot as plt
    ensure_dirs()
    paths = []
    
    # Save the figure first
    for fmt in formats:
        p = CHARTS_DIR / f"{fig_name}.{fmt}"
        plt.savefig(p, dpi=FIGURE_DPI, bbox_inches="tight")
        paths.append(p)
        
    for p in paths:
        print(f"  saved: {p.name}")
        
    # Let matplotlib handle displaying AND safely clearing the memory
    if show:
        plt.show() 
    else:
        plt.close()
        
    return paths


# =============================================================================
# 8. CLIENT ARTIFACT NAMING
# =============================================================================

def client_dir(window_length):
    """Directory holding client .npz artifacts for a given window length."""
    return CLIENTS_DIR / f"L{window_length}"


def client_path(window_length, client_id, source_name):
    """Canonical .npz path for one client at one window length."""
    return client_dir(window_length) / f"client_{client_id:02d}_{source_name}.npz"


# =============================================================================
# 9. CONFIG SUMMARY
# =============================================================================

def config_signature():
    """
    Serializable snapshot of every setting that affects results. Notebooks
    write this alongside their outputs so a cached artifact can be checked
    against the config that produced it.
    """
    return {
        "window_lengths": WINDOW_LENGTHS,
        "primary_window_length": PRIMARY_WINDOW_LENGTH,
        "window_stride": WINDOW_STRIDE,
        "split_ratios": [TRAIN_RATIO, VAL_RATIO, TEST_RATIO],
        "min_windows_per_split": MIN_WINDOWS_PER_SPLIT,
        "log_transform": LOG_TRANSFORM,
        "encoder_hidden_dims": ENCODER_HIDDEN_DIMS,
        "latent_dim": LATENT_DIM,
        "fed_rounds": FED_ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "total_epoch_budget": TOTAL_EPOCH_BUDGET,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "seeds": SEEDS,
        "internet2_routers": INTERNET2_ROUTERS,
        "cesnet_primary_ids": CESNET_PRIMARY_IDS,
        "cesnet_trim_hours": CESNET_TRIM_HOURS,
    }


def print_config_summary():
    print("=" * 72)
    print("INDIS 2026 FEDERATED LEARNING -- ACTIVE CONFIGURATION")
    print("=" * 72)
    print(f"  Window lengths          : {WINDOW_LENGTHS} (primary = {PRIMARY_WINDOW_LENGTH})")
    print(f"  Split ratios            : {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  Min windows per split   : {MIN_WINDOWS_PER_SPLIT}")
    for L in WINDOW_LENGTHS:
        print(f"    -> min hours for L={L:<3d}: {min_hours_required(L)} "
              f"(DERIVED, not hardcoded)")
    print(f"  Log transform           : {LOG_TRANSFORM}")
    print(f"  Architecture            : L -> {' -> '.join(map(str, ENCODER_HIDDEN_DIMS))} "
          f"-> {LATENT_DIM} -> {' -> '.join(map(str, reversed(ENCODER_HIDDEN_DIMS)))} -> L")
    print(f"  Fed rounds x local epochs: {FED_ROUNDS} x {LOCAL_EPOCHS} "
          f"= {TOTAL_EPOCH_BUDGET} epoch budget (DERIVED)")
    print(f"  Seeds                   : {SEEDS}")
    print(f"  Internet2 clients       : {len(INTERNET2_ROUTERS)}")
    print(f"  CESNET clients          : {len(CESNET_PRIMARY_IDS)} "
          f"(trimmed to {CESNET_TRIM_HOURS}h)")
    print(f"  ISP                    : {ISP_ROLE} (NOT a training client)")
    print("=" * 72)


if __name__ == "__main__":
    print_config_summary()
    verify_paths()