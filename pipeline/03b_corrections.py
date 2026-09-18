# %% [markdown]
# # Notebook 03b -- Corrected Figures 3, 4 and Tables 3, 4
#
# Regenerates from CACHED artifacts only. No retraining. Runs in seconds.
#
# Fixes three bugs found in the first pass of Notebook 03:
#
# **Bug 1 (Fig. 4):** group labels inverted. `sorted()` orders network_class
# alphabetically and "CESNET3" < "Internet2", so CESNET clients come first --
# but the label code assumed Internet2 was on the left. Every client was
# labelled with the wrong network.
#
# **Bug 2 (Fig. 3):** the local-only mean suffered survivorship bias. Clients
# early-stop at different epochs; once a client stopped it left the average,
# so the curve tracked only the still-training (i.e. not-yet-converged)
# clients. The result dropped BELOW centralized and federated, contradicting
# Table 3 where local-only is clearly worst. Fixed by forward-filling each
# trajectory's running-best value after it stops -- which is what the operator
# would actually deploy -- so all 20 clients stay in the average at every step.
# Also: federated is now plotted on an EQUAL-COMPUTE x-axis (round x local
# epochs), since one FedAvg round consumes LOCAL_EPOCHS epochs of local work.
#
# **Bug 3 (Table 3):** the reported std pooled variance across clients AND
# seeds, so it measured client heterogeneity (Boston vs. Dallas differ
# enormously), not run-to-run variance. Corrected to: mean across clients
# within each seed, then std across seeds.

# %%
# =============================================================================
# CELL 0 -- Load cached artifacts
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

cfg.ensure_dirs()
cfg.setup_matplotlib()

hist = pd.read_parquet(cfg.METRICS_DIR / "histories.parquet")
pcm = pd.read_parquet(cfg.METRICS_DIR / "per_client_metrics.parquet")
with open(cfg.METRICS_DIR / "comm_costs.json") as f:
    comm = json.load(f)

print("=" * 72)
print("NOTEBOOK 03b -- CORRECTED FIGURES AND TABLES")
print("=" * 72)
print(f"  History rows        : {len(hist):,}")
print(f"  Per-client metrics  : {len(pcm):,}")
print(f"  Seeds               : {sorted(pcm['seed'].unique())}")
print("=" * 72)

# %%
# =============================================================================
# CELL 1 -- Bug 2 fix: survivorship-free convergence curves
# =============================================================================

def running_best_filled(hist_df, regime, max_step):
    """
    Build a (n_trajectories, max_step+1) matrix of running-best val loss,
    forward-filled past each trajectory's early-stop point.

    A trajectory is one (seed) for Centralized/Federated, or one
    (seed, client_id) for Local-only. After early stopping, a trajectory's
    value is held at its best-so-far -- the model the operator would deploy.
    Without this, trajectories silently leave the average when they stop,
    biasing the mean toward whichever runs happen to still be training.
    """
    h = hist_df[hist_df["regime"] == regime].copy()
    if h.empty:
        return None

    keys = ["seed", "client_id"] if regime == "Local-only" else ["seed"]
    if regime == "Local-only" and "client_id" not in h.columns:
        return None

    h = h.sort_values(keys + ["step"])
    h["best"] = h.groupby(keys)["val_loss"].cummin()

    rows = []
    full_index = pd.RangeIndex(0, max_step + 1)
    for _, g in h.groupby(keys):
        s = g.set_index("step")["best"].reindex(full_index).ffill()
        # any leading NaN (trajectory started late) -> backfill so the row is complete
        s = s.bfill()
        rows.append(s.values)
    return np.vstack(rows)


# EQUAL-COMPUTE x-axis: one FedAvg round == LOCAL_EPOCHS epochs of local work.
fed_steps = int(hist[hist["regime"] == "Federated"]["step"].max())
cen_steps = int(hist[hist["regime"] == "Centralized"]["step"].max())
loc_steps = int(hist[hist["regime"] == "Local-only"]["step"].max())
fed_epoch_equiv = (fed_steps + 1) * cfg.LOCAL_EPOCHS
MAX_EPOCH_AXIS = max(cen_steps, loc_steps, fed_epoch_equiv)

print(f"\n  Centralized max epoch      : {cen_steps}")
print(f"  Local-only max epoch       : {loc_steps}")
print(f"  Federated rounds           : {fed_steps + 1} "
      f"(= {fed_epoch_equiv} local epochs of compute)")

curves = {}
for regime in ["Centralized", "Local-only"]:
    M = running_best_filled(hist, regime, MAX_EPOCH_AXIS)
    if M is not None:
        curves[regime] = {"x": np.arange(MAX_EPOCH_AXIS + 1), "M": M}

M_fed = running_best_filled(hist, "Federated", fed_steps)
if M_fed is not None:
    curves["Federated"] = {
        "x": (np.arange(fed_steps + 1) + 1) * cfg.LOCAL_EPOCHS,  # equal-compute
        "M": M_fed,
    }

fig, ax = plt.subplots(figsize=(5.6, 3.8))

for regime in ["Centralized", "Federated"]:
    if regime not in curves:
        continue
    x, M = curves[regime]["x"], curves[regime]["M"]
    mean, std = M.mean(axis=0), M.std(axis=0)
    ax.plot(x, mean, color=cfg.REGIME_COLORS[regime], linewidth=2, label=regime)
    ax.fill_between(x, mean - std, mean + std,
                    color=cfg.REGIME_COLORS[regime], alpha=0.18)

if "Local-only" in curves:
    x, M = curves["Local-only"]["x"], curves["Local-only"]["M"]
    ax.plot(x, M.mean(axis=0), color=cfg.REGIME_COLORS["Local-only"],
            linewidth=2, linestyle="--", label="Local-only (mean)")
    ax.fill_between(x, M.min(axis=0), M.max(axis=0),
                    color=cfg.REGIME_COLORS["Local-only"], alpha=0.15,
                    label="Local-only (client min-max)")

ax.set_yscale("log")
ax.set_xlabel("Local training epochs (FedAvg: round x local epochs)")
ax.set_ylabel("Validation reconstruction MSE (running best)")
ax.set_title("Figure 3: Convergence at equal compute budget", fontsize=11, loc="left")
ax.grid(True, which="both", linestyle=":", alpha=0.45)
ax.legend(fontsize=8)
plt.tight_layout()
cfg.save_figure("F3_convergence")

for regime in ["Centralized", "Federated", "Local-only"]:
    if regime in curves:
        final = curves[regime]["M"][:, -1]
        print(f"  {regime:12s} final val MSE: {final.mean():.5f} "
              f"(+/- {final.std():.5f} across trajectories)")

# %%
# =============================================================================
# CELL 2 -- Bug 1 fix: Figure 4 with CORRECT group labels
# =============================================================================
meta = (pcm[["client_id", "source_name", "network_class"]]
        .drop_duplicates().set_index("client_id"))

# Sort Internet2 FIRST explicitly -- do not rely on alphabetical ordering of
# the class name, which is what inverted the labels in the first pass.
NET_ORDER = {"Internet2": 0, "CESNET3": 1}
order = sorted(meta.index,
               key=lambda c: (NET_ORDER[meta.loc[c, "network_class"]],
                              meta.loc[c, "source_name"]))
labels = [meta.loc[c, "source_name"] for c in order]
net_of = [meta.loc[c, "network_class"] for c in order]

agg = (pcm.groupby(["client_id", "regime"])["test_mae"]
          .agg(["mean", "std"]).reset_index())

x = np.arange(len(order))
width = 0.26
fig, ax = plt.subplots(figsize=(7.4, 3.5))

for i, regime in enumerate(cfg.REGIMES):
    sub = agg[agg["regime"] == regime].set_index("client_id")
    means = [sub.loc[c, "mean"] if c in sub.index else np.nan for c in order]
    errs = [sub.loc[c, "std"] if c in sub.index else 0.0 for c in order]
    ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=2,
           label=regime, color=cfg.REGIME_COLORS[regime],
           edgecolor="black", linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Test reconstruction MAE")
ax.set_title("Figure 4: Per-client reconstruction error "
             "(mean +/- std over seeds)", fontsize=11, loc="left")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=8, loc="upper right")

# Boundary and labels derived from the ACTUAL ordering
boundary = next((i for i, n in enumerate(net_of) if n != net_of[0]), len(net_of))
ax.axvline(boundary - 0.5, color="gray", linewidth=1, alpha=0.7)
ytop = ax.get_ylim()[1]
ax.text(boundary / 2 - 0.5, ytop * 0.96, net_of[0],
        ha="center", fontsize=8, color="gray")
if boundary < len(net_of):
    ax.text(boundary + (len(net_of) - boundary) / 2 - 0.5, ytop * 0.96,
            net_of[boundary], ha="center", fontsize=8, color="gray")

plt.tight_layout()
cfg.save_figure("F4_per_client")

print(f"\n  Left group : {net_of[0]} (clients 0..{boundary - 1})")
print(f"  Right group: {net_of[boundary] if boundary < len(net_of) else 'n/a'}")
print(f"  Order      : {labels}")

# %%
# =============================================================================
# CELL 3 -- Bug 3 fix: Table 3 with a proper seed-level error bar
# =============================================================================
# Mean across clients WITHIN each seed, then std across seeds. The previous
# version pooled clients and seeds, so its std measured client heterogeneity
# (Boston vs. Dallas), not run-to-run variance.
per_seed = (pcm.groupby(["regime", "seed"])[["test_mse", "test_mae"]]
              .mean().reset_index())

t3 = (per_seed.groupby("regime")[["test_mse", "test_mae"]]
        .agg(["mean", "std"]).reset_index())
t3.columns = ["regime", "mse_mean", "mse_std", "mae_mean", "mae_std"]

cen_mse = float(t3.loc[t3["regime"] == "Centralized", "mse_mean"].iloc[0])
t3["error_ratio_vs_centralized"] = (t3["mse_mean"] / cen_mse).round(4)
t3["quality_retained_pct"] = (cen_mse / t3["mse_mean"] * 100).round(1)

t3.to_csv(cfg.METRICS_DIR / "table3_performance.csv", index=False)

print("\nTABLE 3 (CORRECTED) -- seed-level error bars\n")
print(t3.to_string(index=False))

# Is the federated-vs-centralized gap inside seed noise?
cen = per_seed[per_seed["regime"] == "Centralized"]["test_mse"].values
fed = per_seed[per_seed["regime"] == "Federated"]["test_mse"].values
diff = fed - cen
pooled = np.sqrt(cen.std(ddof=1) ** 2 + fed.std(ddof=1) ** 2)

print(f"\n  Centralized per-seed MSE : {np.round(cen, 5)}")
print(f"  Federated   per-seed MSE : {np.round(fed, 5)}")
print(f"  Mean difference (fed-cen): {diff.mean():+.6f}")
print(f"  Pooled seed std          : {pooled:.6f}")
if abs(diff.mean()) < pooled:
    print("\n  -> The gap is WITHIN seed noise. Claim 'federated MATCHES")
    print("     centralized', not 'outperforms'. This is still the strong")
    print("     result: federation recovers full centralized quality.")
else:
    print("\n  -> The gap exceeds pooled seed std. A directional claim may be")
    print("     defensible, but report the per-seed values alongside it.")

# %%
# =============================================================================
# CELL 4 -- Table 4 with explicit granularity qualifiers
# =============================================================================
# The original single "Centralized" row invited the reading "centralization is
# ~1800x cheaper, full stop" -- true only at hourly-aggregate granularity.
# Figure 7 exists precisely to show that this flips at scale, so the table now
# names the granularity and includes the raw-telemetry case.
raw_bytes_per_client = 1.5e9
K = comm["n_clients"]

t4 = pd.DataFrame([
    {"Regime": "Centralized (hourly aggregates)",
     "Total bytes moved": comm["centralize_hourly_total_bytes"],
     "Raw data exposed": "Yes", "Crosses admin boundary": "Yes"},
    {"Regime": "Centralized (raw NetFlow)",
     "Total bytes moved": K * raw_bytes_per_client,
     "Raw data exposed": "Yes", "Crosses admin boundary": "Yes"},
    {"Regime": "Federated (FedAvg weights)",
     "Total bytes moved": comm["fedavg_total_bytes"],
     "Raw data exposed": "No", "Crosses admin boundary": "No"},
    {"Regime": "Local-only (no collaboration)",
     "Total bytes moved": 0,
     "Raw data exposed": "No", "Crosses admin boundary": "No"},
])
t4["Total bytes moved"] = t4["Total bytes moved"].map(lambda v: f"{v:,.0f}")
t4.to_csv(cfg.METRICS_DIR / "table4_communication.csv", index=False)

print("\nTABLE 4 (CORRECTED) -- granularity made explicit\n")
print(t4.to_string(index=False))

print("\n" + "=" * 72)
print("NOTEBOOK 03b COMPLETE")
print("=" * 72)
print("  Regenerated: F3_convergence, F4_per_client,")
print("               table3_performance.csv, table4_communication.csv")
print("  Unchanged  : F1, F5, F6, F7, F8 (no bugs found)")
print("=" * 72)
