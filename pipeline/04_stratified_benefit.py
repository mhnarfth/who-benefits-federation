# %% [markdown]
# # Notebook 04 -- Who Benefits from Federation?
#
# The paper's central finding. Runs from cached artifacts in seconds.
#
# Diagnostic 3 showed federation wins on 15/20 clients. Splitting those wins
# by network class reveals the benefit is NOT uniform: Internet2 clients gain
# an order of magnitude more than CESNET3 clients, and this survives removing
# the Boston outlier.
#
# It also corrects a confound: the r = -0.620 correlation between training-set
# size and federation benefit is largely a BETWEEN-GROUP artifact, since every
# CESNET client has exactly 1035 windows (the maximum) while every data-poor
# client is Internet2. This notebook tests the correlation WITHIN each network
# separately, which is the only way to separate data quantity from regime.
#
# **Outputs:** `F9_who_benefits`, `F10_val_test_gap`,
# `table5_stratified_benefit.csv`, `stats_tests.json`

# %%
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

cfg.ensure_dirs()
cfg.setup_matplotlib()

pcm = pd.read_parquet(cfg.METRICS_DIR / "per_client_metrics.parquet")
hist = pd.read_parquet(cfg.METRICS_DIR / "histories.parquet")

per_client = (pcm.groupby(["regime", "client_id", "source_name", "network_class"])
                 [["test_mse", "test_mae"]].mean().reset_index())
sizes = pcm.groupby("source_name")["n_train"].first()

wide = per_client.pivot_table(index=["source_name", "network_class"],
                              columns="regime", values="test_mae").reset_index()
wide["benefit_pct"] = 100 * (wide["Local-only"] - wide["Federated"]) / wide["Local-only"]
wide["n_train"] = wide["source_name"].map(sizes)
wide = wide.sort_values("benefit_pct", ascending=False)

print("=" * 78)
print("STRATIFIED FEDERATION BENEFIT")
print("=" * 78)

# %%
# =============================================================================
# CELL 1 -- Benefit stratified by network class
# =============================================================================
strat = (wide.groupby("network_class")["benefit_pct"]
             .agg(["count", "mean", "median", "min", "max"]).round(2))
print("\nBenefit by network class (federated vs local-only, test MAE):\n")
print(strat.to_string())

i2 = wide[wide.network_class == "Internet2"]["benefit_pct"].values
ces = wide[wide.network_class == "CESNET3"]["benefit_pct"].values

# Robustness: does the finding survive removing the biggest winners?
i2_sorted = np.sort(i2)[::-1]
print(f"\n  Internet2 mean:            {i2.mean():+.1f}%")
print(f"  Internet2 mean (drop top 1): {i2_sorted[1:].mean():+.1f}%")
print(f"  Internet2 mean (drop top 2): {i2_sorted[2:].mean():+.1f}%")
print(f"  CESNET3   mean:            {ces.mean():+.1f}%")
print(f"\n  Internet2 MEDIAN: {np.median(i2):+.1f}%   "
      f"CESNET3 MEDIAN: {np.median(ces):+.1f}%")
print("  The median is immune to Boston -- if the gap persists here,")
print("  the finding is structural, not outlier-driven.")

# %%
# =============================================================================
# CELL 2 -- Statistical tests
# =============================================================================
tests = {}

# (a) Is federation better than local-only across clients? Paired, non-parametric.
w_stat, w_p = stats.wilcoxon(wide["Local-only"], wide["Federated"],
                             alternative="greater")
tests["wilcoxon_fed_vs_local"] = {
    "statistic": float(w_stat), "p_value": float(w_p),
    "n_clients": int(len(wide)),
    "interpretation": "paired one-sided: local-only error > federated error",
}

# (b) Do Internet2 clients benefit more than CESNET clients? Independent groups.
u_stat, u_p = stats.mannwhitneyu(i2, ces, alternative="greater")
tests["mannwhitney_i2_vs_cesnet_benefit"] = {
    "statistic": float(u_stat), "p_value": float(u_p),
    "i2_median": float(np.median(i2)), "cesnet_median": float(np.median(ces)),
    "interpretation": "one-sided: Internet2 benefit > CESNET3 benefit",
}

print("\n" + "=" * 78)
print("STATISTICAL TESTS")
print("=" * 78)
print(f"\n  (a) Wilcoxon signed-rank, federated vs local-only across 20 clients")
print(f"      p = {w_p:.5f}  {'SIGNIFICANT' if w_p < 0.05 else 'not significant'} at alpha=0.05")
print(f"\n  (b) Mann-Whitney U, Internet2 benefit vs CESNET3 benefit")
print(f"      Internet2 median {np.median(i2):+.1f}%  vs  CESNET3 median {np.median(ces):+.1f}%")
print(f"      p = {u_p:.5f}  {'SIGNIFICANT' if u_p < 0.05 else 'not significant'} at alpha=0.05")

# %%
# =============================================================================
# CELL 3 -- Decompose the data-size confound
# =============================================================================
# The pooled r = -0.620 is suspect: all CESNET clients sit at exactly 1035
# windows while every data-poor client is Internet2. Test WITHIN each group.
print("\n" + "=" * 78)
print("IS IT DATA SIZE, OR IS IT NETWORK REGIME?")
print("=" * 78)

r_pooled = stats.pearsonr(wide["n_train"], wide["benefit_pct"])
print(f"\n  Pooled across all 20 clients : r = {r_pooled[0]:+.3f} (p = {r_pooled[1]:.4f})")

for net in ["Internet2", "CESNET3"]:
    g = wide[wide.network_class == net]
    if g["n_train"].nunique() < 3:
        print(f"  Within {net:10s}          : n_train is near-constant "
              f"({g['n_train'].nunique()} unique values) -- correlation undefined")
        tests[f"pearson_within_{net}"] = {"note": "n_train near-constant, not computable"}
    else:
        r = stats.pearsonr(g["n_train"], g["benefit_pct"])
        print(f"  Within {net:10s}          : r = {r[0]:+.3f} (p = {r[1]:.4f})")
        tests[f"pearson_within_{net}"] = {"r": float(r[0]), "p_value": float(r[1])}

tests["pearson_pooled"] = {"r": float(r_pooled[0]), "p_value": float(r_pooled[1])}

# The clean counterexample: matched data size, opposite outcome.
print("\n  MATCHED-SIZE COUNTEREXAMPLE (data quantity does not order these):\n")
cmp_rows = wide[wide["source_name"].isin(["louisville", "inst0"])]
print(cmp_rows[["source_name", "network_class", "n_train", "benefit_pct"]]
      .to_string(index=False))
print("\n  Near-identical training-set size, opposite sign of benefit.")
print("  Argue REGIME (machine- vs human-driven traffic), not data volume.")

# %%
# =============================================================================
# CELL 4 -- Figure 9: the who-benefits figure (paper's centrepiece)
# =============================================================================
fig, ax = plt.subplots(figsize=(6.2, 4.0))

colors = [cfg.NETWORK_COLORS[n] for n in wide["network_class"]]
y = np.arange(len(wide))
ax.barh(y, wide["benefit_pct"], color=colors, edgecolor="black", linewidth=0.4)
ax.set_yticks(y)
ax.set_yticklabels(wide["source_name"], fontsize=7.5)
ax.invert_yaxis()
ax.axvline(0, color="black", linewidth=0.9)

for net in ["Internet2", "CESNET3"]:
    med = np.median(wide[wide.network_class == net]["benefit_pct"])
    ax.axvline(med, color=cfg.NETWORK_COLORS[net], linestyle="--",
               linewidth=1.4, alpha=0.85,
               label=f"{net} median ({med:+.1f}%)")

ax.set_xlabel("Federation benefit: reduction in test MAE vs local-only (%)")
ax.set_title("Figure 9: Federation benefit is stratified by network regime",
             fontsize=11, loc="left")
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
cfg.save_figure("F9_who_benefits")

# %%
# =============================================================================
# CELL 5 -- Figure 10: validation-to-test generalization gap
# =============================================================================
# Local-only achieves the BEST validation loss but the WORST test loss. Each
# silo early-stops on its own 95-204 window validation set, so the model
# SELECTION itself overfits. Centralized and federated select against ~3,700
# pooled windows and generalize far better.
rows = []
for regime in cfg.REGIMES:
    h = hist[hist["regime"] == regime]
    if h.empty:
        continue
    keys = ["seed", "client_id"] if (regime == "Local-only" and "client_id" in h.columns) else ["seed"]
    rows.append({
        "regime": regime,
        "val": float(h.groupby(keys)["val_loss"].min().mean()),
        "test": float(pcm[pcm.regime == regime]["test_mse"].mean()),
    })
vt = pd.DataFrame(rows)
vt["degradation"] = vt["test"] / vt["val"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))

x = np.arange(len(vt))
w = 0.36
ax1.bar(x - w / 2, vt["val"], w, label="Validation (best)",
        color="#9DB4D0", edgecolor="black", linewidth=0.4)
ax1.bar(x + w / 2, vt["test"], w, label="Test",
        color="#2F4B7C", edgecolor="black", linewidth=0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(vt["regime"], fontsize=8, rotation=12)
ax1.set_ylabel("Reconstruction MSE")
ax1.set_title("A: Validation vs test error", fontsize=10, loc="left")
ax1.legend(fontsize=8)
ax1.grid(axis="y", linestyle=":", alpha=0.5)

bar_colors = [cfg.REGIME_COLORS[r] for r in vt["regime"]]
ax2.bar(x, vt["degradation"], 0.55, color=bar_colors,
        edgecolor="black", linewidth=0.4)
for i, v in enumerate(vt["degradation"]):
    ax2.text(i, v + 0.05, f"{v:.2f}x", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(1.0, color="gray", linestyle=":", linewidth=1)
ax2.set_xticks(x)
ax2.set_xticklabels(vt["regime"], fontsize=8, rotation=12)
ax2.set_ylabel("Test / validation error ratio")
ax2.set_title("B: Generalization gap", fontsize=10, loc="left")
ax2.set_ylim(0, max(vt["degradation"]) * 1.25)
ax2.grid(axis="y", linestyle=":", alpha=0.5)

plt.suptitle("Figure 10: Federation improves model-selection reliability",
             fontsize=11, y=1.02)
plt.tight_layout()
cfg.save_figure("F10_val_test_gap")

print("\n" + vt.round(4).to_string(index=False))

# %%
# =============================================================================
# CELL 6 -- Table 5 and saved statistics
# =============================================================================
t5 = wide[["source_name", "network_class", "n_train",
           "Centralized", "Federated", "Local-only", "benefit_pct"]].copy()
t5.columns = ["Client", "Network", "Train windows",
              "Centralized MAE", "Federated MAE", "Local-only MAE", "Benefit (%)"]
t5.to_csv(cfg.METRICS_DIR / "table5_stratified_benefit.csv", index=False)

tests["stratified_summary"] = {
    "internet2_mean": float(i2.mean()), "internet2_median": float(np.median(i2)),
    "cesnet_mean": float(ces.mean()), "cesnet_median": float(np.median(ces)),
    "internet2_mean_drop_top1": float(np.sort(i2)[::-1][1:].mean()),
    "internet2_mean_drop_top2": float(np.sort(i2)[::-1][2:].mean()),
    "wins": int((wide["benefit_pct"] > 2).sum()),
    "ties": int((wide["benefit_pct"].abs() <= 2).sum()),
    "losses": int((wide["benefit_pct"] < -2).sum()),
}
with open(cfg.METRICS_DIR / "stats_tests.json", "w") as f:
    json.dump(tests, f, indent=2)

print("\n" + "=" * 78)
print("HEADLINE NUMBERS FOR THE PAPER")
print("=" * 78)
print(f"  Federation wins/ties/loses     : "
      f"{tests['stratified_summary']['wins']}/"
      f"{tests['stratified_summary']['ties']}/"
      f"{tests['stratified_summary']['losses']} of 20 clients")
print(f"  Internet2 median benefit       : {np.median(i2):+.1f}%")
print(f"  CESNET3 median benefit         : {np.median(ces):+.1f}%")
print(f"  Wilcoxon (fed > local)         : p = {w_p:.5f}")
print(f"  Mann-Whitney (I2 > CESNET)     : p = {u_p:.5f}")
print(f"  Local-only val->test degradation: "
      f"{float(vt[vt.regime == 'Local-only']['degradation'].iloc[0]):.2f}x")
print(f"  Federated val->test degradation : "
      f"{float(vt[vt.regime == 'Federated']['degradation'].iloc[0]):.2f}x")
print("=" * 78)
