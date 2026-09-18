# %% [markdown]
# # Diagnostic -- Is the local-only result driven by one client?
#
# Two questions this settles from cached artifacts (no retraining, seconds):
#
# 1. **Leave-one-out sensitivity.** Table 3 reports local-only as 71% worse
#    than federated. Figure 4 suggests that gap may be dominated by Boston
#    alone. If so, the honest headline is "federation rescues data-poor
#    silos", NOT "federation improves every client", and the paper must say
#    the former.
#
# 2. **Val-to-test generalization gap.** Local-only has the BEST validation
#    loss and the WORST test loss. If that holds up it is a real finding:
#    small silos overfit their own model selection, and federation fixes it.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

pcm = pd.read_parquet(cfg.METRICS_DIR / "per_client_metrics.parquet")
hist = pd.read_parquet(cfg.METRICS_DIR / "histories.parquet")

print("=" * 78)
print("DIAGNOSTIC 1 -- MEAN vs MEDIAN (outlier sensitivity)")
print("=" * 78)

per_client = (pcm.groupby(["regime", "client_id", "source_name", "network_class"])
                 [["test_mse", "test_mae"]].mean().reset_index())

summary = (per_client.groupby("regime")[["test_mse", "test_mae"]]
             .agg(["mean", "median"]).round(4))
print(summary.to_string())

fed_mean = per_client[per_client.regime == "Federated"]["test_mse"].mean()
loc_mean = per_client[per_client.regime == "Local-only"]["test_mse"].mean()
fed_med = per_client[per_client.regime == "Federated"]["test_mse"].median()
loc_med = per_client[per_client.regime == "Local-only"]["test_mse"].median()

print(f"\n  Local-only vs Federated (MSE)")
print(f"    by MEAN   : {loc_mean:.4f} vs {fed_mean:.4f}  "
      f"-> local-only {100 * (loc_mean / fed_mean - 1):+.1f}%")
print(f"    by MEDIAN : {loc_med:.4f} vs {fed_med:.4f}  "
      f"-> local-only {100 * (loc_med / fed_med - 1):+.1f}%")
print("\n  If MEAN and MEDIAN disagree sharply, the mean is outlier-driven")
print("  and the paper must report the median (or both).")

# %%
print("\n" + "=" * 78)
print("DIAGNOSTIC 2 -- LEAVE-ONE-CLIENT-OUT SENSITIVITY")
print("=" * 78)
print("  Recomputes the local-only vs federated gap with each client removed.")
print("  A single client whose removal collapses the gap IS the result.\n")

rows = []
for drop in sorted(per_client["client_id"].unique()):
    sub = per_client[per_client["client_id"] != drop]
    f = sub[sub.regime == "Federated"]["test_mse"].mean()
    l = sub[sub.regime == "Local-only"]["test_mse"].mean()
    name = per_client[per_client.client_id == drop]["source_name"].iloc[0]
    rows.append({"dropped": name, "fed_mse": f, "local_mse": l,
                 "gap_pct": 100 * (l / f - 1)})

loo = pd.DataFrame(rows).sort_values("gap_pct")
full_gap = 100 * (loc_mean / fed_mean - 1)
print(f"  Full-federation gap (all 20 clients): local-only {full_gap:+.1f}% worse\n")
print(f"  {'dropped client':<16s} {'gap without it':>16s} {'change':>12s}")
print("  " + "-" * 48)
for _, r in loo.iterrows():
    print(f"  {r['dropped']:<16s} {r['gap_pct']:>15.1f}% {r['gap_pct'] - full_gap:>11.1f}")

worst = loo.iloc[0]
print(f"\n  Most influential client: {worst['dropped']}")
print(f"    gap falls from {full_gap:+.1f}% to {worst['gap_pct']:+.1f}% when removed")
if worst["gap_pct"] < full_gap * 0.5:
    print(f"\n  -> The aggregate IS dominated by '{worst['dropped']}'. Report the")
    print(f"     finding as ROBUSTNESS/EQUITY (federation rescues data-poor")
    print(f"     silos) rather than a uniform accuracy improvement.")
else:
    print(f"\n  -> The gap is broadly distributed, not one-client-driven.")
    print(f"     A general accuracy claim is defensible.")

# %%
print("\n" + "=" * 78)
print("DIAGNOSTIC 3 -- PER-CLIENT WIN / TIE / LOSS")
print("=" * 78)
print("  'Federated matches or beats local-only on N of 20 clients' is a more")
print("  precise and more defensible claim than any single mean ratio.\n")

piv = per_client.pivot(index="source_name", columns="regime", values="test_mae")
piv["delta"] = piv["Local-only"] - piv["Federated"]
piv["fed_better_pct"] = 100 * piv["delta"] / piv["Local-only"]
piv = piv.sort_values("fed_better_pct", ascending=False)

TIE = 0.02  # within 2% counts as a tie
wins = int((piv["fed_better_pct"] > TIE * 100).sum())
ties = int((piv["fed_better_pct"].abs() <= TIE * 100).sum())
loss = int((piv["fed_better_pct"] < -TIE * 100).sum())

print(f"  {'client':<16s} {'federated':>10s} {'local-only':>11s} {'fed better by':>14s}")
print("  " + "-" * 56)
for name, r in piv.iterrows():
    print(f"  {name:<16s} {r['Federated']:>10.4f} {r['Local-only']:>11.4f} "
          f"{r['fed_better_pct']:>13.1f}%")

print(f"\n  Federated WINS : {wins}/20   TIES: {ties}/20   LOSES: {loss}/20")
print(f"  (tie band = +/-{TIE * 100:.0f}%)")

# %%
print("\n" + "=" * 78)
print("DIAGNOSTIC 4 -- VALIDATION-TO-TEST GENERALIZATION GAP")
print("=" * 78)
print("  Local-only had the BEST val loss but the WORST test loss. If the")
print("  degradation ratio is much larger for local-only, that is model-")
print("  SELECTION overfitting: each silo early-stops on its own small val")
print("  set, while centralized/federated select against pooled validation.\n")

val_rows = []
for regime in cfg.REGIMES:
    h = hist[hist["regime"] == regime]
    if h.empty:
        continue
    keys = ["seed", "client_id"] if (regime == "Local-only" and "client_id" in h.columns) else ["seed"]
    best_val = h.groupby(keys)["val_loss"].min().mean()
    test_mse = pcm[pcm.regime == regime]["test_mse"].mean()
    val_rows.append({"regime": regime, "best_val_mse": best_val,
                     "test_mse": test_mse, "degradation": test_mse / best_val})

vt = pd.DataFrame(val_rows)
print(vt.round(4).to_string(index=False))

if len(vt) == 3:
    loc_deg = float(vt[vt.regime == "Local-only"]["degradation"].iloc[0])
    fed_deg = float(vt[vt.regime == "Federated"]["degradation"].iloc[0])
    print(f"\n  Local-only degrades {loc_deg:.2f}x from val to test")
    print(f"  Federated  degrades {fed_deg:.2f}x")
    if loc_deg > fed_deg * 1.3:
        print("\n  -> CONFIRMED. Federation improves model-selection reliability,")
        print("     not merely training. This is a distinct, publishable claim")
        print("     and connects directly to temporal generalization.")

# %%
print("\n" + "=" * 78)
print("DIAGNOSTIC 5 -- DOES CLIENT DATA SIZE PREDICT THE FEDERATION BENEFIT?")
print("=" * 78)

sizes = pcm.groupby("source_name")["n_train"].first()
merged = piv.join(sizes)
corr = merged["n_train"].corr(merged["fed_better_pct"])

print(f"  {'client':<16s} {'train windows':>14s} {'fed better by':>14s}")
print("  " + "-" * 48)
for name, r in merged.sort_values("n_train").iterrows():
    print(f"  {name:<16s} {int(r['n_train']):>14d} {r['fed_better_pct']:>13.1f}%")

print(f"\n  Pearson r (train size vs federation benefit) = {corr:.3f}")
if corr < -0.4:
    print("  -> Data-poor clients benefit MORE. The equity story is supported")
    print("     by a real correlation, not just the Boston anecdote.")
elif abs(corr) <= 0.4:
    print("  -> Weak correlation. The benefit is NOT simply a data-quantity")
    print("     effect; do not claim it is without a stronger predictor.")
print("=" * 78)
