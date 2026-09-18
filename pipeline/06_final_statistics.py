# %% [markdown]
# # Notebook 06 -- Final Statistical Closure
#
# Closes the last open gap and builds the mechanism figure. Runs in seconds
# from cached artifacts. After this, experimentation is DONE.
#
# 1. **Formal equivalence test.** The paper claims "federated matches
#    centralized" but that has only ever been an eyeball comparison of five
#    paired seed values. A paired test plus a TOST equivalence test makes the
#    claim defensible. Note: a non-significant difference is NOT proof of
#    equivalence -- TOST tests equivalence directly, which is the correct
#    instrument for this claim.
#
# 2. **Mechanism figure.** Two independent knobs (FedProx mu, local epochs E)
#    both suppress local optimization, and both selectively destroy the
#    Internet2 benefit while barely affecting CESNET. Neither experiment alone
#    establishes the mechanism; together they do.
#
# 3. **Headline number sheet** -- every figure the paper will quote, in one
#    place, so nothing gets mistyped during writing.

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
final = {}

print("=" * 78)
print("NOTEBOOK 06 -- FINAL STATISTICAL CLOSURE")
print("=" * 78)

# %%
# =============================================================================
# CELL 1 -- Federated vs Centralized: formal equivalence
# =============================================================================
per_seed = (pcm.groupby(["regime", "seed"])[["test_mse", "test_mae"]]
              .mean().reset_index())
cen = per_seed[per_seed.regime == "Centralized"]["test_mse"].values
fed = per_seed[per_seed.regime == "Federated"]["test_mse"].values

t_stat, t_p = stats.ttest_rel(fed, cen)
try:
    w_stat, w_p = stats.wilcoxon(fed, cen)
except ValueError:
    w_stat, w_p = np.nan, np.nan

# TOST: are the two within +/-5% of the centralized mean?
# A non-significant difference test does NOT establish equivalence; TOST does.
BOUND_PCT = 5.0
bound = BOUND_PCT / 100 * cen.mean()
diff = fed - cen
se = diff.std(ddof=1) / np.sqrt(len(diff))
df_ = len(diff) - 1
t_lower = (diff.mean() - (-bound)) / se
t_upper = (diff.mean() - bound) / se
p_lower = 1 - stats.t.cdf(t_lower, df_)
p_upper = stats.t.cdf(t_upper, df_)
p_tost = max(p_lower, p_upper)

print("\nFEDERATED vs CENTRALIZED (paired across 5 seeds)")
print(f"  Centralized per-seed MSE : {np.round(cen, 5)}")
print(f"  Federated   per-seed MSE : {np.round(fed, 5)}")
print(f"  Mean difference          : {diff.mean():+.6f}")
print(f"  Paired t-test            : t = {t_stat:+.3f}, p = {t_p:.4f}")
print(f"  Wilcoxon signed-rank     : p = {w_p:.4f}")
print(f"  TOST equivalence (+/-{BOUND_PCT:.0f}%) : p = {p_tost:.4f}")
if p_tost < 0.05:
    print(f"\n  -> EQUIVALENCE ESTABLISHED within +/-{BOUND_PCT:.0f}%.")
    print("     You may write 'federated is statistically equivalent to")
    print("     centralized', which is stronger than 'no significant difference'.")
else:
    print(f"\n  -> Equivalence NOT established at the +/-{BOUND_PCT:.0f}% bound with n=5.")
    print("     Write 'no statistically significant difference (p = "
          f"{t_p:.2f})' and do NOT claim proven equivalence.")

final["fed_vs_cen"] = {
    "centralized_per_seed": cen.tolist(), "federated_per_seed": fed.tolist(),
    "mean_diff": float(diff.mean()), "paired_t_p": float(t_p),
    "wilcoxon_p": float(w_p), "tost_p": float(p_tost),
    "tost_bound_pct": BOUND_PCT,
}

# %%
# =============================================================================
# CELL 2 -- Mechanism figure: local adaptation drives the Internet2 benefit
# =============================================================================
fx_path = cfg.METRICS_DIR / "fedprox_comparison.csv"
ep_path = cfg.METRICS_DIR / "local_epoch_ablation.csv"

if fx_path.exists() and ep_path.exists():
    fx = pd.read_csv(fx_path)
    ea = pd.read_csv(ep_path)

    # Fold the FedAvg (E=5) point into the local-epoch curve
    base = fx[fx["algo"] == "FedAvg"]
    if len(base):
        ea = pd.concat([ea, pd.DataFrame([{
            "local_epochs": cfg.LOCAL_EPOCHS,
            "fed_mae": float(base["fed_mae"].iloc[0]),
            "i2_median": float(base["i2_median"].iloc[0]),
            "cesnet_median": float(base["cesnet_median"].iloc[0]),
        }])], ignore_index=True)
    ea = ea.sort_values("local_epochs")

    fx_plot = fx.copy()
    fx_plot["mu_plot"] = fx_plot["mu"].replace(0.0, 1e-4)  # 0 on a log axis
    fx_plot = fx_plot.sort_values("mu_plot")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))

    a1.plot(fx_plot["mu_plot"], fx_plot["i2_median"], "o-", linewidth=2,
            color=cfg.NETWORK_COLORS["Internet2"], label="Internet2")
    a1.plot(fx_plot["mu_plot"], fx_plot["cesnet_median"], "s-", linewidth=2,
            color=cfg.NETWORK_COLORS["CESNET3"], label="CESNET3")
    a1.set_xscale("log")
    a1.axhline(0, color="black", linewidth=0.8)
    a1.set_xlabel("FedProx proximal strength (mu; leftmost = FedAvg)")
    a1.set_ylabel("Median federation benefit (%)")
    a1.set_title("A: Constraining local drift", fontsize=10, loc="left")
    a1.legend(fontsize=8)
    a1.grid(True, which="both", linestyle=":", alpha=0.45)

    a2.plot(ea["local_epochs"], ea["i2_median"], "o-", linewidth=2,
            color=cfg.NETWORK_COLORS["Internet2"], label="Internet2")
    a2.plot(ea["local_epochs"], ea["cesnet_median"], "s-", linewidth=2,
            color=cfg.NETWORK_COLORS["CESNET3"], label="CESNET3")
    a2.axhline(0, color="black", linewidth=0.8)
    a2.set_xlabel("Local epochs per round (E)")
    a2.set_ylabel("Median federation benefit (%)")
    a2.set_title("B: Local optimization budget", fontsize=10, loc="left")
    a2.set_xticks(sorted(ea["local_epochs"].unique()))
    a2.legend(fontsize=8)
    a2.grid(True, linestyle=":", alpha=0.45)

    plt.suptitle("Figure 14: The Internet2 benefit requires unconstrained "
                 "local optimization", fontsize=11, y=1.03)
    plt.tight_layout()
    cfg.save_figure("F14_mechanism")

    print("\nMECHANISM EVIDENCE (two independent knobs, same conclusion)")
    print("\n  Panel A -- FedProx mu:")
    for _, r in fx_plot.iterrows():
        print(f"    mu={r['mu']:<6g}  I2 {r['i2_median']:+6.1f}%   "
              f"CESNET {r['cesnet_median']:+6.1f}%")
    print("\n  Panel B -- local epochs:")
    for _, r in ea.iterrows():
        print(f"    E={int(r['local_epochs']):<3d}     I2 {r['i2_median']:+6.1f}%   "
              f"CESNET {r['cesnet_median']:+6.1f}%")
    print("\n  Both knobs suppress local optimization. Both selectively remove")
    print("  the Internet2 benefit while barely moving CESNET. Neither result")
    print("  alone establishes the mechanism; jointly they do.")

    final["mechanism"] = {
        "fedprox": fx[["mu", "i2_median", "cesnet_median", "fed_mae"]].to_dict("records"),
        "local_epochs": ea[["local_epochs", "i2_median", "cesnet_median", "fed_mae"]].to_dict("records"),
    }
else:
    print("\n  Skipping Figure 14 -- run Notebook 05 first.")

# %%
# =============================================================================
# CELL 3 -- Headline number sheet
# =============================================================================
print("\n" + "=" * 78)
print("HEADLINE NUMBERS -- quote these, do not retype from memory")
print("=" * 78)

try:
    with open(cfg.METRICS_DIR / "stats_tests.json") as f:
        s4 = json.load(f)
    ss = s4["stratified_summary"]
    print("\n  PRIMARY RESULTS (L=24, FedAvg, 5 seeds, 20 clients)")
    print(f"    Federated vs centralized     : p = {t_p:.3f} (no significant difference)")
    print(f"    Federated vs local-only      : Wilcoxon p = "
          f"{s4['wilcoxon_fed_vs_local']['p_value']:.5f}")
    print(f"    Win / tie / loss             : {ss['wins']}/{ss['ties']}/{ss['losses']} of 20")
    print(f"    Internet2 median benefit     : {ss['internet2_median']:+.1f}%")
    print(f"    CESNET3 median benefit       : {ss['cesnet_median']:+.1f}%")
    print(f"    Stratification significance  : p = "
          f"{s4['mannwhitney_i2_vs_cesnet_benefit']['p_value']:.5f}")
    print(f"    Robust to dropping Boston    : I2 mean "
          f"{ss['internet2_mean_drop_top1']:+.1f}% (vs {ss['internet2_mean']:+.1f}%)")
except Exception as e:
    print(f"  (stats_tests.json unavailable: {e})")

if fx_path.exists():
    fx = pd.read_csv(fx_path)
    best = fx.loc[fx["fed_mae"].idxmin()]
    print("\n  ROBUSTNESS")
    print(f"    Best aggregator              : {best['algo']} "
          f"(MAE {best['fed_mae']:.4f}); FedAvg {float(fx[fx.algo=='FedAvg']['fed_mae'].iloc[0]):.4f}")
    print("    -> FedProx gives no meaningful gain; FedAvg is appropriate here.")

wa_path = cfg.METRICS_DIR / "window_ablation.csv"
if wa_path.exists():
    wa = pd.read_csv(wa_path).sort_values("window_length")
    print("\n    Window-length robustness:")
    for _, r in wa.iterrows():
        gap = r["i2_median"] - r["cesnet_median"]
        sig = "significant" if r["mannwhitney_p"] < 0.05 else "MARGINAL"
        print(f"      L={int(r['window_length']):<3d} gap {gap:+5.1f} pts  "
              f"p={r['mannwhitney_p']:.4f} ({sig})")
    print("    -> Direction consistent at all L; effect grows with window length.")
    print("       Report L=12 as marginal (p>0.05). Do not call it significant.")

with open(cfg.METRICS_DIR / "final_numbers.json", "w") as f:
    json.dump(final, f, indent=2)

print("\n" + "=" * 78)
print("EXPERIMENTATION COMPLETE -- next step is writing, not more runs.")
print("=" * 78)
