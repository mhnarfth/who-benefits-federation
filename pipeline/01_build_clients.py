# %% [markdown]
# # Notebook 01 -- Data Preparation and Federated Client Construction
#
# **INDIS 2026:** *Learning Without Centralizing: Federated Representation
# Learning for Multi-Network Traffic Telemetry*
#
# **Inputs:** Internet2 hourly Parquet (10 routers), CESNET3 hourly CSV
# (10 institutions), ISP ISP snapshot.
#
# **Outputs:**
# - `outputs/indis2026/clients/L{12,24,48}/client_XX_<name>.npz`
# - `outputs/indis2026/metrics/table1_clients.csv`  (Table 1)
# - `outputs/indis2026/charts/F1_traffic_regimes.{pdf,png}`  (Figure 1)
# - `outputs/indis2026/metrics/notebook01_manifest.json`
#
# **Critical ordering:** split chronologically -> fit scaler on TRAIN ONLY ->
# window within each split independently. Windowing before splitting would
# leak 23 of 24 hours between adjacent train/test windows.
#
# This file uses `# %%` cell markers: open directly as a notebook in VSCode or
# JupyterLab (jupytext), or run as a plain script.

# %%
# =============================================================================
# CELL 0 -- Imports and path verification (fail loudly before any work)
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# fed_config.py sits at the project root, one level above notebooks/
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

cfg.ensure_dirs()
cfg.setup_matplotlib()
cfg.print_config_summary()

path_status = cfg.verify_paths(verbose=True)
isp_available = path_status["ISP_PATH"]["exists"]
if not isp_available:
    print("\n  NOTE: ISP path not found. The federation (20 training clients)")
    print("        does not depend on ISP -- it is a holdout probe only.")
    print("        Set cfg.ISP_PATH to enable the unseen-network transfer result.")

# %%
# =============================================================================
# CELL 1 -- ISP DIAGNOSTIC
# =============================================================================
# ISP's schema is unknown to this pipeline. Inspect whatever is actually
# there and report it, rather than assuming a format and silently mis-parsing.
# After running this, set ISP_TIME_COL / ISP_VALUE_COL / ISP_IS_PREAGGREGATED
# in fed_config.py if auto-detection below does not pick the right columns.

def diagnose_isp(path):
    print("=" * 72)
    print("ISP DIAGNOSTIC")
    print("=" * 72)
    if not path.exists():
        print(f"  Path does not exist: {path}")
        return None

    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.is_file()])
        print(f"  Directory with {len(files)} file(s):")
        for f in files[:20]:
            print(f"    {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
        candidates = [f for f in files if f.suffix in (".parquet", ".csv", ".pq")]
        if not candidates:
            print("  No .parquet/.csv files found.")
            return None
        target = candidates[0]
        print(f"\n  Inspecting first candidate: {target.name}")
    else:
        target = path
        print(f"  Single file: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")

    df = None
    for reader, label in [(pd.read_parquet, "parquet"), (pd.read_csv, "csv")]:
        try:
            df = reader(target)
            print(f"  Parsed as {label}.")
            break
        except Exception as e:
            print(f"  Not {label}: {type(e).__name__}")

    if df is None:
        print("  Could not parse. Inspect manually.")
        return None

    print(f"\n  Shape   : {df.shape}")
    print(f"  Columns : {df.columns.tolist()}")
    print(f"\n  Dtypes:\n{df.dtypes.to_string()}")
    print(f"\n  Head:\n{df.head().to_string()}")

    n_rows = len(df)
    print(f"\n  INTERPRETATION HINT:")
    if n_rows < 200:
        print(f"    {n_rows} rows -> looks PRE-AGGREGATED (hourly). "
              f"Expected ~49 rows for a 2-day snapshot.")
    else:
        print(f"    {n_rows:,} rows -> looks like RAW FLOWS. "
              f"Will need hourly aggregation (expect ~49 hourly points).")
    print("=" * 72)
    return df


isp_probe_df = diagnose_isp(cfg.ISP_PATH) if isp_available else None

# %%
# =============================================================================
# CELL 2 -- Loader functions
# =============================================================================

def load_internet2_series(router):
    """Internet2 hourly series. Reads ONLY `in_packets`.

    The if_score / if_flag anomaly columns are physically present in these
    files but are deliberately NOT read: this paper is univariate by design,
    which keeps it cleanly separated from the anomaly-aware forecasting work.
    """
    fp = cfg.INTERNET2_DIR / f"hourly_{router}_processed_with_anomalies.parquet"
    if not fp.exists():
        raise FileNotFoundError(f"Internet2 file missing: {fp}")
    df = pd.read_parquet(fp, columns=["in_packets"])
    s = df["in_packets"].interpolate(method="linear").fillna(0.0)
    return s.values.astype(np.float64), df.index


def load_cesnet_times():
    """id_time -> timestamp lookup. Timestamps are tz-aware UTC; strip tz."""
    t = pd.read_csv(cfg.CESNET_TIMES_PATH)
    t["time"] = pd.to_datetime(t["time"], utc=True).dt.tz_localize(None)
    return t


def load_cesnet_series(inst_id, times_df):
    """CESNET institution hourly series, trimmed to CESNET_TRIM_HOURS."""
    fp = cfg.CESNET_INSTITUTIONS_DIR / f"{inst_id}.csv"
    if not fp.exists():
        raise FileNotFoundError(f"CESNET file missing: {fp}")
    df = pd.read_csv(fp, usecols=["id_time", "n_packets"])
    df = df.merge(times_df, on="id_time", how="left").sort_values("time")
    df = df.set_index("time")
    s = df["n_packets"].interpolate(method="linear").fillna(0.0)
    if len(s) > cfg.CESNET_TRIM_HOURS:
        s = s.iloc[:cfg.CESNET_TRIM_HOURS]
    return s.values.astype(np.float64), s.index


def load_isp_series(df):
    """ISP hourly series, with column auto-detection and raw-flow fallback.

    Honest note: this loader is written defensively because ISP's schema was
    not known when this pipeline was designed. If auto-detection picks wrong,
    set ISP_TIME_COL / ISP_VALUE_COL explicitly in fed_config.py.
    """
    if df is None:
        return None, None

    time_col = cfg.ISP_TIME_COL
    value_col = cfg.ISP_VALUE_COL

    if time_col is None:
        for cand in ["time", "timestamp", "t_first", "datetime", "hour", "id_time"]:
            if cand in df.columns:
                time_col = cand
                break
    if value_col is None:
        for cand in ["in_packets", "n_packets", "packets", "pkts", "value"]:
            if cand in df.columns:
                value_col = cand
                break

    if time_col is None or value_col is None:
        raise ValueError(
            f"Could not auto-detect ISP columns. Found: {df.columns.tolist()}. "
            f"Set cfg.ISP_TIME_COL and cfg.ISP_VALUE_COL explicitly."
        )
    print(f"  ISP columns -> time: '{time_col}', value: '{value_col}'")

    d = df[[time_col, value_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
    d = d.dropna(subset=[time_col]).set_index(time_col).sort_index()

    is_preagg = cfg.ISP_IS_PREAGGREGATED
    if is_preagg is None:
        is_preagg = len(d) < 200  # ~49 hourly points vs. millions of raw flows

    if is_preagg:
        s = d[value_col]
        print(f"  ISP treated as PRE-AGGREGATED ({len(s)} points).")
    else:
        s = d[value_col].resample("h").sum()
        print(f"  ISP aggregated from {len(d):,} raw flows -> {len(s)} hourly points.")

    s = s.interpolate(method="linear").fillna(0.0)
    return s.values.astype(np.float64), s.index

# %%
# =============================================================================
# CELL 3 -- Load all raw series
# =============================================================================
raw_series = {}   # client_id -> dict(values, index, network_class, source_name)
client_id = 0

print("\nLoading Internet2 routers...")
for router in cfg.INTERNET2_ROUTERS:
    vals, idx = load_internet2_series(router)
    raw_series[client_id] = {
        "values": vals, "index": idx,
        "network_class": "Internet2", "source_name": router,
    }
    print(f"  [{client_id:02d}] Internet2/{router:14s} {len(vals):5d} h")
    client_id += 1

print("\nLoading CESNET3 institutions...")
cesnet_times = load_cesnet_times()
for inst_id in cfg.CESNET_PRIMARY_IDS:
    vals, idx = load_cesnet_series(inst_id, cesnet_times)
    name = f"inst{inst_id}"
    raw_series[client_id] = {
        "values": vals, "index": idx,
        "network_class": "CESNET3", "source_name": name,
    }
    print(f"  [{client_id:02d}] CESNET3/{name:16s} {len(vals):5d} h")
    client_id += 1

isp_entry = None
if isp_available and isp_probe_df is not None:
    print("\nLoading ISP (holdout probe)...")
    vals, idx = load_isp_series(isp_probe_df)
    if vals is not None:
        isp_entry = {
            "values": vals, "index": idx,
            "network_class": "ISP", "source_name": "isp",
        }
        print(f"  [--] ISP/isp                  {len(vals):5d} h  (HOLDOUT PROBE)")

print(f"\nTraining clients: {len(raw_series)}   Holdout probes: {1 if isp_entry else 0}")

# %%
# =============================================================================
# CELL 4 -- Windowing and normalization
# =============================================================================

def make_windows(x, L, stride=1):
    """Stride-`stride` sliding windows. Returns shape (n_windows, L)."""
    if len(x) < L:
        return np.empty((0, L), dtype=np.float32)
    starts = np.arange(0, len(x) - L + 1, stride)
    return np.stack([x[i:i + L] for i in starts]).astype(np.float32)


def split_normalize_window(series, L, stride=None, log_transform=None):
    """
    Split chronologically -> fit scaler on TRAIN ONLY -> window each split.

    Per-client normalization is not merely hygiene: a GLOBAL normalizer would
    require sharing statistics across clients, which is precisely what the
    federated setting forbids. It also destroys magnitude information, so any
    downstream network-class separation reflects temporal SHAPE, not scale --
    a materially stronger claim.
    """
    if stride is None:
        stride = cfg.WINDOW_STRIDE
    if log_transform is None:
        log_transform = cfg.LOG_TRANSFORM

    T = len(series)
    n_tr, n_va, n_te = cfg.split_sizes(T)
    raw = {
        "train": series[:n_tr],
        "val":   series[n_tr:n_tr + n_va],
        "test":  series[n_tr + n_va:],
    }

    x_tr = np.log1p(raw["train"]) if log_transform else raw["train"]
    mu = float(np.mean(x_tr))
    sd = float(np.std(x_tr))
    if sd < 1e-8:
        sd = 1.0  # degenerate constant series guard

    windows, hours = {}, {}
    for name, arr in raw.items():
        x = np.log1p(arr) if log_transform else arr
        windows[name] = make_windows((x - mu) / sd, L, stride)
        hours[name] = len(arr)

    scaler = {"mu": mu, "sigma": sd, "log_transform": bool(log_transform)}
    return windows, scaler, hours


def window_full_series(series, L, scaler_mu=None, scaler_sigma=None,
                       stride=None, log_transform=None):
    """Window an ENTIRE series without splitting -- used for the ISP probe."""
    if stride is None:
        stride = cfg.WINDOW_STRIDE
    if log_transform is None:
        log_transform = cfg.LOG_TRANSFORM

    x = np.log1p(series) if log_transform else series
    if scaler_mu is None:
        scaler_mu = float(np.mean(x))
    if scaler_sigma is None:
        scaler_sigma = float(np.std(x))
        if scaler_sigma < 1e-8:
            scaler_sigma = 1.0
    return (make_windows((x - scaler_mu) / scaler_sigma, L, stride),
            {"mu": scaler_mu, "sigma": scaler_sigma, "log_transform": bool(log_transform)})

# %%
# =============================================================================
# CELL 5 -- Build and save clients for every window length
# =============================================================================
inventory = []

for L in cfg.WINDOW_LENGTHS:
    outdir = cfg.client_dir(L)
    outdir.mkdir(parents=True, exist_ok=True)
    min_h = cfg.min_hours_required(L)
    print(f"\n=== L = {L:2d}  (min hours required = {min_h}, DERIVED) ===")

    for cid, entry in raw_series.items():
        T = len(entry["values"])
        eligible = T >= min_h
        if not eligible:
            print(f"  [{cid:02d}] {entry['source_name']:16s} SKIPPED "
                  f"({T} h < {min_h} h)")
            continue

        windows, scaler, hours = split_normalize_window(entry["values"], L)
        fp = cfg.client_path(L, cid, entry["source_name"])
        np.savez_compressed(
            fp,
            train=windows["train"], val=windows["val"], test=windows["test"],
            scaler_mu=scaler["mu"], scaler_sigma=scaler["sigma"],
            log_transform=scaler["log_transform"],
            client_id=cid, network_class=entry["network_class"],
            source_name=entry["source_name"], role=cfg.TRAIN_CLIENT_ROLE,
            window_length=L, stride=cfg.WINDOW_STRIDE,
            total_hours=T, train_hours=hours["train"],
            val_hours=hours["val"], test_hours=hours["test"],
        )
        inventory.append({
            "window_length": L, "client_id": cid,
            "network_class": entry["network_class"],
            "source_name": entry["source_name"],
            "role": cfg.TRAIN_CLIENT_ROLE, "total_hours": T,
            "train_hours": hours["train"], "val_hours": hours["val"],
            "test_hours": hours["test"],
            "train_windows": len(windows["train"]),
            "val_windows": len(windows["val"]),
            "test_windows": len(windows["test"]),
        })
        print(f"  [{cid:02d}] {entry['source_name']:16s} "
              f"{T:5d} h -> windows train/val/test = "
              f"{len(windows['train']):5d}/{len(windows['val']):4d}/{len(windows['test']):4d}")

    # ISP: whole series windowed, no split, flagged as holdout probe
    if isp_entry is not None:
        T = len(isp_entry["values"])
        probe_w, probe_scaler = window_full_series(isp_entry["values"], L)
        fp = outdir / "probe_isp.npz"
        np.savez_compressed(
            fp,
            probe=probe_w,
            train=np.empty((0, L), dtype=np.float32),
            val=np.empty((0, L), dtype=np.float32),
            test=np.empty((0, L), dtype=np.float32),
            scaler_mu=probe_scaler["mu"], scaler_sigma=probe_scaler["sigma"],
            log_transform=probe_scaler["log_transform"],
            client_id=-1, network_class="ISP", source_name="isp",
            role=cfg.ISP_ROLE, window_length=L, stride=cfg.WINDOW_STRIDE,
            total_hours=T,
        )
        inventory.append({
            "window_length": L, "client_id": -1, "network_class": "ISP",
            "source_name": "isp", "role": cfg.ISP_ROLE, "total_hours": T,
            "train_hours": 0, "val_hours": 0, "test_hours": 0,
            "train_windows": 0, "val_windows": 0, "test_windows": 0,
            "probe_windows": len(probe_w),
        })
        print(f"  [--] isp (HOLDOUT PROBE)  {T:5d} h -> {len(probe_w)} probe windows")

inv_df = pd.DataFrame(inventory)
print(f"\nWrote {len(inv_df)} client artifacts across {len(cfg.WINDOW_LENGTHS)} window lengths.")

# %%
# =============================================================================
# CELL 6 -- Hard assertions
# =============================================================================
# Silent data problems are far more expensive than a loud failure here.
print("\n" + "=" * 72)
print("VALIDATION")
print("=" * 72)

primary = inv_df[(inv_df["window_length"] == cfg.PRIMARY_WINDOW_LENGTH) &
                 (inv_df["role"] == cfg.TRAIN_CLIENT_ROLE)]

assert len(primary) > 0, "No training clients built at the primary window length."

for _, r in primary.iterrows():
    for split in ["train", "val", "test"]:
        n = r[f"{split}_windows"]
        assert n >= cfg.MIN_WINDOWS_PER_SPLIT, (
            f"Client {r['client_id']} ({r['source_name']}) has only {n} "
            f"{split} windows (< {cfg.MIN_WINDOWS_PER_SPLIT})."
        )
print(f"  [OK] All {len(primary)} training clients meet the "
      f"{cfg.MIN_WINDOWS_PER_SPLIT}-window minimum in every split.")

isp_rows = inv_df[inv_df["network_class"] == "ISP"]
if len(isp_rows):
    assert (isp_rows["role"] == cfg.ISP_ROLE).all(), "ISP mis-flagged as a training client."
    assert (isp_rows["train_windows"] == 0).all(), "ISP has training windows -- must be 0."
    print("  [OK] ISP is flagged holdout_probe with zero training windows.")

nan_found = False
for L in cfg.WINDOW_LENGTHS:
    for fp in sorted(cfg.client_dir(L).glob("*.npz")):
        d = np.load(fp, ispw_pickle=True)
        for key in ["train", "val", "test", "probe"]:
            if key in d and d[key].size and not np.isfinite(d[key]).all():
                print(f"  [FAIL] Non-finite values in {fp.name}/{key}")
                nan_found = True
assert not nan_found, "Non-finite values detected in client artifacts."
print("  [OK] No NaN/Inf in any client artifact.")

n_train_clients = len(primary)
print(f"\n  Federation size at L={cfg.PRIMARY_WINDOW_LENGTH}: K = {n_train_clients} training clients")
print(f"  Internet2: {(primary['network_class'] == 'Internet2').sum()}   "
      f"CESNET3: {(primary['network_class'] == 'CESNET3').sum()}   "
      f"ISP: holdout probe")
print("=" * 72)

# %%
# =============================================================================
# CELL 7 -- Table 1 (client / federation statistics)
# =============================================================================
tbl = inv_df[inv_df["window_length"] == cfg.PRIMARY_WINDOW_LENGTH].copy()

rows = []
for net in ["Internet2", "CESNET3", "ISP"]:
    g = tbl[tbl["network_class"] == net]
    if not len(g):
        continue
    is_probe = net == "ISP"
    regime = {"Internet2": "machine-driven",
              "CESNET3": "human-driven",
              "ISP": "broadband"}[net]
    rows.append({
        "Network": net,
        "Clients": len(g),
        "Role": "holdout probe" if is_probe else "training client",
        "Regime": regime,
        "Hours (min-max)": (f"{int(g['total_hours'].min())}" if is_probe else
                            f"{int(g['total_hours'].min())}-{int(g['total_hours'].max())}"),
        "Train windows (min-max)": ("n/a" if is_probe else
                                    f"{int(g['train_windows'].min())}-{int(g['train_windows'].max())}"),
        "Test windows (min-max)": ("n/a" if is_probe else
                                   f"{int(g['test_windows'].min())}-{int(g['test_windows'].max())}"),
    })

table1 = pd.DataFrame(rows)
table1_path = cfg.METRICS_DIR / "table1_clients.csv"
table1.to_csv(table1_path, index=False)

per_client_path = cfg.METRICS_DIR / "table1_per_client_detail.csv"
tbl.to_csv(per_client_path, index=False)

print("\nTABLE 1 -- Federation composition\n")
print(table1.to_string(index=False))
print(f"\n  saved: {table1_path.name}")
print(f"  saved: {per_client_path.name}")

# %%
# =============================================================================
# CELL 8 -- Figure 1 (three traffic regimes)
# =============================================================================
# Panels 1-2 span 168 h (a full week, so CESNET's weekend dip is visible).
# Panel 3 spans only ISP's ~49 h. THE X-RANGES DIFFER BY NECESSITY -- the
# paper caption must state this so no reader assumes a shared axis.

WEEK_HOURS = 168


def normalize_for_plot(x):
    x = np.log1p(x)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


def pick(net, name):
    for cid, e in raw_series.items():
        if e["network_class"] == net and e["source_name"] == name:
            return e
    for cid, e in raw_series.items():
        if e["network_class"] == net:
            return e
    return None


i2 = pick("Internet2", "dallas")
cn = pick("CESNET3", "inst2")

n_panels = 3 if isp_entry is not None else 2
fig, axes = plt.subplots(n_panels, 1, figsize=(7.0, 2.1 * n_panels))

# Panel 1 -- Internet2 (machine-driven)
seg = i2["values"][:WEEK_HOURS]
axes[0].plot(np.arange(len(seg)), normalize_for_plot(seg),
             color=cfg.NETWORK_COLORS["Internet2"], linewidth=1.3)
axes[0].set_title(f"Internet2 backbone router ({i2['source_name']}) - machine-driven",
                  fontsize=10, loc="left")
axes[0].set_ylabel("Norm. volume")
axes[0].set_xlim(0, WEEK_HOURS)

# Panel 2 -- CESNET3 (human-driven)
seg = cn["values"][:WEEK_HOURS]
axes[1].plot(np.arange(len(seg)), normalize_for_plot(seg),
             color=cfg.NETWORK_COLORS["CESNET3"], linewidth=1.3)
axes[1].set_title(f"CESNET3 institution ({cn['source_name']}) - human-driven",
                  fontsize=10, loc="left")
axes[1].set_ylabel("Norm. volume")
axes[1].set_xlim(0, WEEK_HOURS)

# Panel 3 -- ISP (broadband, shorter axis)
if isp_entry is not None:
    seg = isp_entry["values"]
    axes[2].plot(np.arange(len(seg)), normalize_for_plot(seg),
                 color=cfg.NETWORK_COLORS["ISP"], linewidth=1.3)
    axes[2].set_title(f"ISP regional ISP - broadband (note: {len(seg)} h axis)",
                      fontsize=10, loc="left")
    axes[2].set_ylabel("Norm. volume")
    axes[2].set_xlim(0, len(seg))

for ax in axes:
    ax.grid(True, linestyle=":", alpha=0.5)
axes[-1].set_xlabel("Hours from start of segment")

plt.suptitle("Figure 1: Three network traffic regimes in the federation",
             fontsize=11, fontweight="bold", y=1.00)
plt.tight_layout()
cfg.save_figure("F1_traffic_regimes")

# %%
# =============================================================================
# CELL 9 -- Manifest and handoff
# =============================================================================
manifest = {
    "notebook": "01_data_clients",
    "config_signature": cfg.config_signature(),
    "n_training_clients": int(n_train_clients),
    "n_internet2_clients": int((primary["network_class"] == "Internet2").sum()),
    "n_cesnet_clients": int((primary["network_class"] == "CESNET3").sum()),
    "isp_available": bool(isp_entry is not None),
    "isp_role": cfg.ISP_ROLE,
    "window_lengths_built": cfg.WINDOW_LENGTHS,
    "primary_window_length": cfg.PRIMARY_WINDOW_LENGTH,
    "min_hours_required": {str(L): cfg.min_hours_required(L) for L in cfg.WINDOW_LENGTHS},
    "artifacts": {
        "clients_dir": str(cfg.CLIENTS_DIR),
        "table1": str(table1_path),
        "table1_detail": str(per_client_path),
        "figure1": str(cfg.CHARTS_DIR / "F1_traffic_regimes.pdf"),
    },
}
manifest_path = cfg.METRICS_DIR / "notebook01_manifest.json"
with open(manifest_path, "w") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 72)
print("NOTEBOOK 01 COMPLETE")
print("=" * 72)
print(f"  Training clients (K)  : {n_train_clients}")
print(f"  Holdout probe         : {'ISP' if isp_entry else 'none'}")
print(f"  Window lengths built  : {cfg.WINDOW_LENGTHS}")
print(f"  Client artifacts      : {cfg.CLIENTS_DIR}")
print(f"  Manifest              : {manifest_path.name}")
print("\n  Next: 02_train_regimes.ipynb "
      "(centralized / federated / local-only)")
print("=" * 72)
