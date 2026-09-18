"""
regen_figure2.py -- Regenerates ONLY the traffic-regime figure.

WHY: the current version plots the FIRST 168 hours of Dallas. That segment
contains scheduled maintenance outages (zero traffic). After min-max
normalisation the outages dominate the range and the panel renders as a
binary square wave -- a picture of maintenance windows, not of the
machine-driven traffic texture the figure exists to show. Since this figure
carries the regime argument that the entire paper depends on, it has to be
legible.

FIX: automatically select the cleanest available 168-hour window (fewest
zero-traffic hours) for each panel, and report the choice so it can be stated
in the caption. Outages are then discussed separately in the text as a REN
characteristic rather than being ispwed to swamp the figure.

DOES NOT regenerate client artifacts. Reads source data, writes only
charts/F2_traffic_regimes.{pdf,png}. Notebooks 02 and 05 are unaffected.

Run from the project root:  python regen_figure2.py
Runtime: a few seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
import fed_config as cfg

cfg.ensure_dirs()
cfg.setup_matplotlib()

WEEK_HOURS = 168
I2_ROUTER = "dallas"
CESNET_INST = "2"


# ---------------------------------------------------------------- loaders --
def load_internet2(router):
    fp = cfg.INTERNET2_DIR / f"hourly_{router}_processed_with_anomalies.parquet"
    df = pd.read_parquet(fp, columns=["in_packets"])
    return df["in_packets"].interpolate(method="linear").fillna(0).values.astype(float)


def load_cesnet(inst_id):
    times = pd.read_csv(cfg.CESNET_TIMES_PATH)
    times["time"] = pd.to_datetime(times["time"], utc=True).dt.tz_localize(None)
    df = pd.read_csv(cfg.CESNET_INSTITUTIONS_DIR / f"{inst_id}.csv",
                     usecols=["id_time", "n_packets"])
    df = df.merge(times, on="id_time", how="left").sort_values("time")
    s = df["n_packets"].interpolate(method="linear").fillna(0)
    if len(s) > cfg.CESNET_TRIM_HOURS:
        s = s.iloc[:cfg.CESNET_TRIM_HOURS]
    return s.values.astype(float)


def load_isp():
    if not cfg.ISP_PATH.exists():
        return None
    df = pd.read_parquet(cfg.ISP_PATH)
    tcol, vcol = cfg.ISP_TIME_COL, cfg.ISP_VALUE_COL
    # start_datetime is the index in this file, not a column
    if tcol not in df.columns and df.index.name == tcol:
        df = df.reset_index()
    d = df[[tcol, vcol]].copy()
    d[tcol] = pd.to_datetime(d[tcol], errors="coerce")
    d = d.dropna(subset=[tcol]).set_index(tcol).sort_index()
    return d[vcol].astype(float).values


# ------------------------------------------------------- window selection --
def cleanest_window(values, window_hours):
    """Return (start_index, segment, n_zeros) for the window with fewest zeros.

    Scans every candidate start position and stops early on the first
    zero-free window. Deterministic: earliest qualifying window wins.
    """
    n = len(values)
    if n <= window_hours:
        return 0, values, int((values == 0).sum())
    best_start, best_zeros = 0, np.inf
    for s in range(0, n - window_hours + 1):
        z = int((values[s:s + window_hours] == 0).sum())
        if z < best_zeros:
            best_zeros, best_start = z, s
        if z == 0:
            break
    return best_start, values[best_start:best_start + window_hours], int(best_zeros)


def normalize_for_plot(x):
    x = np.log1p(x)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)


# ------------------------------------------------------------------- main --
def main():
    print("=" * 70)
    print("REGENERATING TRAFFIC-REGIME FIGURE")
    print("=" * 70)

    i2_full = load_internet2(I2_ROUTER)
    cn_full = load_cesnet(CESNET_INST)
    isp_full = load_isp()

    i2_zeros_total = int((i2_full == 0).sum())
    first_window_zeros = int((i2_full[:WEEK_HOURS] == 0).sum())

    i2_start, i2_seg, i2_z = cleanest_window(i2_full, WEEK_HOURS)
    cn_start, cn_seg, cn_z = cleanest_window(cn_full, WEEK_HOURS)

    print(f"\n  Internet2/{I2_ROUTER}: {len(i2_full)} h total, "
          f"{i2_zeros_total} zero-traffic hours overall")
    print(f"    OLD window (hours 0-{WEEK_HOURS})   : {first_window_zeros} zeros "
          f"<- this is what broke the panel")
    print(f"    NEW window (hours {i2_start}-{i2_start + WEEK_HOURS}) : {i2_z} zeros")
    print(f"\n  CESNET3/inst{CESNET_INST}: {len(cn_full)} h total")
    print(f"    NEW window (hours {cn_start}-{cn_start + WEEK_HOURS}) : {cn_z} zeros")
    if isp_full is not None:
        print(f"\n  ISP: {len(isp_full)} h total, "
              f"{int((isp_full == 0).sum())} zero-traffic hours")

    if i2_z > 0:
        print(f"\n  WARNING: no zero-free 168h window exists for {I2_ROUTER}.")
        print(f"           Best available still has {i2_z} zeros. Consider a")
        print(f"           different router (edit I2_ROUTER at the top).")

    # ------------------------------------------------------------- figure --
    n_panels = 3 if isp_full is not None else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(7.0, 2.1 * n_panels))

    axes[0].plot(np.arange(len(i2_seg)), normalize_for_plot(i2_seg),
                 color=cfg.NETWORK_COLORS["Internet2"], linewidth=1.3)
    axes[0].set_title(f"Internet2 backbone router ({I2_ROUTER}) - machine-driven",
                      fontsize=10, loc="left")
    axes[0].set_ylabel("Norm. volume")
    axes[0].set_xlim(0, WEEK_HOURS)

    axes[1].plot(np.arange(len(cn_seg)), normalize_for_plot(cn_seg),
                 color=cfg.NETWORK_COLORS["CESNET3"], linewidth=1.3)
    axes[1].set_title(f"CESNET3 institution (inst{CESNET_INST}) - human-driven",
                      fontsize=10, loc="left")
    axes[1].set_ylabel("Norm. volume")
    axes[1].set_xlim(0, WEEK_HOURS)

    if isp_full is not None:
        axes[2].plot(np.arange(len(isp_full)), normalize_for_plot(isp_full),
                     color=cfg.NETWORK_COLORS["ISP"], linewidth=1.3)
        axes[2].set_title(f"ISP regional ISP - broadband "
                          f"(note: {len(isp_full)} h axis)",
                          fontsize=10, loc="left")
        axes[2].set_ylabel("Norm. volume")
        axes[2].set_xlim(0, len(isp_full))

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.5)
    axes[-1].set_xlabel("Hours from start of segment")

    plt.suptitle("Three network traffic regimes in the federation",
                 fontsize=11, fontweight="bold", y=1.00)
    plt.tight_layout()
    cfg.save_figure("F2_traffic_regimes")

    print("\n" + "=" * 70)
    print("CAPTION FACTS (state these in the paper)")
    print("=" * 70)
    print(f"  Internet2 panel : 168 h beginning at hour {i2_start} of the "
          f"{I2_ROUTER} series")
    print(f"  CESNET3 panel   : 168 h beginning at hour {cn_start}")
    if isp_full is not None:
        print(f"  ISP panel      : full {len(isp_full)} h series "
              f"(SHORTER X-AXIS -- say so in the caption)")
    print(f"\n  Also state in the text: the {I2_ROUTER} series contains "
          f"{i2_zeros_total} zero-traffic")
    print("  hours corresponding to scheduled maintenance windows, a known REN")
    print("  characteristic; the plotted segment excludes them for legibility.")
    print("=" * 70)


if __name__ == "__main__":
    main()
