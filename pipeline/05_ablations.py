# %% [markdown]
# # Notebook 05 -- FedProx and Ablations
#
# Closes the three most likely reviewer objections:
#
# 1. **"Why only FedAvg? Your clients are extreme non-IID."** -> FedProx with a
#    mu sweep. Any outcome is reportable: a tie means simple averaging
#    suffices; a win is a better result; a loss means the proximal term
#    over-constrains local adaptation under this much heterogeneity.
#
# 2. **"Does your central finding survive a different window length?"** ->
#    Retrain all regimes at L = 12 and L = 48 and recompute the stratified
#    benefit. Notebook 01 already built these client artifacts, so this costs
#    only compute. THIS IS THE HIGHEST-VALUE CHECK: it tests the paper's main
#    claim, not a side detail.
#
# 3. **"How does local-epoch count trade off against communication?"** ->
#    E in {1, 5, 10} at fixed rounds. Feeds the Figure 7 crossover argument.
#
# Estimated total runtime on an H200: ~1 hour.
#
# **Outputs:** `metrics/ablation_results.parquet`,
# `metrics/fedprox_comparison.csv`, `metrics/window_ablation.csv`,
# `F11_fedprox`, `F12_window_ablation`, `F13_local_epochs`

# %%
# =============================================================================
# CELL 0 -- Setup
# =============================================================================
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

cfg.ensure_dirs()
cfg.setup_matplotlib()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- What to run (set False to skip a block) ---
RUN_FEDPROX = True
RUN_WINDOW_ABLATION = True
RUN_LOCAL_EPOCH_ABLATION = True

FEDPROX_MUS = [0.001, 0.01, 0.1, 1.0]
ABLATION_WINDOWS = [12, 48]          # L=24 already trained in Notebook 02
ABLATION_LOCAL_EPOCHS = [1, 10]      # E=5 already trained in Notebook 02
ABLATION_SEEDS = cfg.SEEDS           # reduce to cfg.SEEDS[:3] if pressed for time

print("=" * 78)
print("NOTEBOOK 05 -- FEDPROX AND ABLATIONS")
print("=" * 78)
print(f"  Device        : {DEVICE}")
print(f"  FedProx mu    : {FEDPROX_MUS if RUN_FEDPROX else 'skipped'}")
print(f"  Window abl.   : {ABLATION_WINDOWS if RUN_WINDOW_ABLATION else 'skipped'}")
print(f"  Local-ep abl. : {ABLATION_LOCAL_EPOCHS if RUN_LOCAL_EPOCH_ABLATION else 'skipped'}")
print(f"  Seeds         : {ABLATION_SEEDS}")
print("=" * 78)

# %%
# =============================================================================
# CELL 1 -- Model and primitives (self-contained; must match Notebook 02)
# =============================================================================

class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims=None, latent_dim=None):
        super().__init__()
        hidden_dims = hidden_dims or cfg.ENCODER_HIDDEN_DIMS
        latent_dim = latent_dim or cfg.LATENT_DIM
        enc, d = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc)
        dec, d = [], latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def load_clients(window_length, device):
    clients = {}
    for fp in sorted(cfg.client_dir(window_length).glob("client_*.npz")):
        d = np.load(fp, ispw_pickle=True)
        cid = int(d["client_id"])
        clients[cid] = {
            "train": torch.tensor(d["train"], dtype=torch.float32, device=device),
            "val":   torch.tensor(d["val"],   dtype=torch.float32, device=device),
            "test":  torch.tensor(d["test"],  dtype=torch.float32, device=device),
            "network_class": str(d["network_class"]),
            "source_name": str(d["source_name"]),
        }
    return clients


@torch.no_grad()
def eval_mse(model, X):
    if X.shape[0] == 0:
        return float("nan")
    model.eval()
    recon, _ = model(X)
    return float(F.mse_loss(recon, X).item())


@torch.no_grad()
def eval_mae(model, X):
    if X.shape[0] == 0:
        return float("nan")
    model.eval()
    recon, _ = model(X)
    return float(torch.mean(torch.abs(recon - X)).item())


def make_gen(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


def run_epochs(model, X, n_epochs, gen, opt=None, global_params=None, mu=0.0):
    """Local training. With mu > 0 and global_params set, this is FedProx:
    the proximal term (mu/2)||theta - theta_global||^2 penalizes drift from
    the round's starting global model."""
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                               weight_decay=cfg.WEIGHT_DECAY)
    n = X.shape[0]
    model.train()
    last = float("nan")
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=X.device, generator=gen)
        running, nb = 0.0, 0
        for i in range(0, n, cfg.BATCH_SIZE):
            xb = X[perm[i:i + cfg.BATCH_SIZE]]
            opt.zero_grad(set_to_none=True)
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            if mu > 0.0 and global_params is not None:
                prox = sum(((p - g) ** 2).sum()
                           for p, g in zip(model.parameters(), global_params))
                loss = loss + 0.5 * mu * prox
            loss.backward()
            opt.step()
            running += float(loss.item())
            nb += 1
        last = running / max(nb, 1)
    return model, opt, last

# %%
# =============================================================================
# CELL 2 -- Federated training (FedAvg when mu=0, FedProx when mu>0)
# =============================================================================

def train_federated(clients, input_dim, seed, mu=0.0, local_epochs=None,
                    rounds=None, verbose=False):
    local_epochs = local_epochs or cfg.LOCAL_EPOCHS
    rounds = rounds or cfg.FED_ROUNDS

    cfg.set_seed(seed)
    global_model = MLPAutoencoder(input_dim).to(DEVICE)

    n_k = {cid: c["train"].shape[0] for cid, c in clients.items()}
    n_total = sum(n_k.values())
    weights = {cid: n_k[cid] / n_total for cid in clients}

    best_val, best_state, best_round = float("inf"), None, -1

    for rnd in range(rounds):
        global_state = copy.deepcopy(global_model.state_dict())
        global_params = [p.detach().clone() for p in global_model.parameters()]
        agg = {k: torch.zeros_like(v, dtype=torch.float32)
               for k, v in global_state.items()}

        for cid, c in clients.items():
            gen = make_gen(seed * 10_000 + rnd * 100 + cid, DEVICE)
            local = MLPAutoencoder(input_dim).to(DEVICE)
            local.load_state_dict(global_state)
            run_epochs(local, c["train"], local_epochs, gen,
                       global_params=global_params, mu=mu)
            w = weights[cid]
            for k, v in local.state_dict().items():
                agg[k] += w * v.float()

        global_model.load_state_dict(agg)
        mean_val = float(np.nanmean([eval_mse(global_model, c["val"])
                                     for c in clients.values()]))
        if mean_val < best_val - 1e-9:
            best_val, best_round = mean_val, rnd
            best_state = copy.deepcopy(global_model.state_dict())

        if verbose and rnd % 25 == 0:
            print(f"      round {rnd:3d}  val {mean_val:.5f}")

    global_model.load_state_dict(best_state)
    return global_model, {"best_round": best_round, "best_val": best_val}


def train_local_only(clients, input_dim, seed):
    models = {}
    for cid, c in clients.items():
        cfg.set_seed(seed + cid)
        gen = make_gen(seed + cid, DEVICE)
        model = MLPAutoencoder(input_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                               weight_decay=cfg.WEIGHT_DECAY)
        best_val, best_state, patience = float("inf"), None, 0
        for _ in range(cfg.LOCAL_ONLY_MAX_EPOCHS):
            model, opt, _ = run_epochs(model, c["train"], 1, gen, opt=opt)
            v = eval_mse(model, c["val"])
            if v < best_val - 1e-9:
                best_val, best_state, patience = v, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
                if patience >= cfg.EARLY_STOP_PATIENCE:
                    break
        model.load_state_dict(best_state)
        models[cid] = model
    return models


def evaluate_all(clients, fed_model, local_models, tag_fields):
    rows = []
    for cid, c in clients.items():
        rows.append({**tag_fields, "client_id": cid,
                     "source_name": c["source_name"],
                     "network_class": c["network_class"], "regime": "Federated",
                     "test_mse": eval_mse(fed_model, c["test"]),
                     "test_mae": eval_mae(fed_model, c["test"])})
        rows.append({**tag_fields, "client_id": cid,
                     "source_name": c["source_name"],
                     "network_class": c["network_class"], "regime": "Local-only",
                     "test_mse": eval_mse(local_models[cid], c["test"]),
                     "test_mae": eval_mae(local_models[cid], c["test"])})
    return rows


def stratified_benefit(df):
    """Median federation benefit per network class, plus the Mann-Whitney p."""
    from scipy import stats as st
    w = df.pivot_table(index=["source_name", "network_class"],
                       columns="regime", values="test_mae").reset_index()
    w["benefit_pct"] = 100 * (w["Local-only"] - w["Federated"]) / w["Local-only"]
    i2 = w[w.network_class == "Internet2"]["benefit_pct"].values
    ces = w[w.network_class == "CESNET3"]["benefit_pct"].values
    p = st.mannwhitneyu(i2, ces, alternative="greater").pvalue if len(i2) and len(ces) else np.nan
    return {"i2_median": float(np.median(i2)), "i2_mean": float(np.mean(i2)),
            "cesnet_median": float(np.median(ces)), "cesnet_mean": float(np.mean(ces)),
            "mannwhitney_p": float(p),
            "wins": int((w["benefit_pct"] > 2).sum()),
            "ties": int((w["benefit_pct"].abs() <= 2).sum()),
            "losses": int((w["benefit_pct"] < -2).sum())}

# %%
# =============================================================================
# CELL 3 -- FedProx mu sweep at the primary window length
# =============================================================================
all_rows, summaries = [], []

if RUN_FEDPROX:
    L = cfg.PRIMARY_WINDOW_LENGTH
    clients = load_clients(L, DEVICE)
    print(f"\n{'=' * 78}\nFEDPROX SWEEP (L={L}, K={len(clients)})\n{'=' * 78}")

    # Reuse the Notebook 02 local-only models as the shared baseline.
    print("  Training local-only baselines (shared across mu values)...")
    t0 = time.time()
    local_by_seed = {s: train_local_only(clients, L, s) for s in ABLATION_SEEDS}
    print(f"    done in {time.time() - t0:.0f}s")

    for mu in [0.0] + FEDPROX_MUS:      # mu=0.0 reproduces FedAvg as control
        algo = "FedAvg" if mu == 0.0 else f"FedProx(mu={mu})"
        print(f"\n  {algo}")
        t0 = time.time()
        rows = []
        for seed in ABLATION_SEEDS:
            fed, info = train_federated(clients, L, seed, mu=mu)
            rows += evaluate_all(clients, fed, local_by_seed[seed],
                                 {"experiment": "fedprox", "algo": algo,
                                  "mu": mu, "window_length": L,
                                  "local_epochs": cfg.LOCAL_EPOCHS, "seed": seed})
        df = pd.DataFrame(rows)
        agg = df.groupby(["source_name", "network_class", "regime"])["test_mae"].mean().reset_index()
        s = stratified_benefit(agg)
        fed_mae = df[df.regime == "Federated"]["test_mae"].mean()
        s.update({"algo": algo, "mu": mu, "fed_mae": fed_mae})
        summaries.append(s)
        all_rows += rows
        print(f"    fed MAE {fed_mae:.4f} | I2 median {s['i2_median']:+.1f}% | "
              f"CESNET median {s['cesnet_median']:+.1f}% | "
              f"W/T/L {s['wins']}/{s['ties']}/{s['losses']} | {time.time() - t0:.0f}s")

    fx = pd.DataFrame(summaries)
    fx.to_csv(cfg.METRICS_DIR / "fedprox_comparison.csv", index=False)
    print(f"\n{fx[['algo', 'fed_mae', 'i2_median', 'cesnet_median', 'mannwhitney_p']].round(4).to_string(index=False)}")

# %%
# =============================================================================
# CELL 4 -- Window-length ablation (robustness of the CENTRAL claim)
# =============================================================================
win_summaries = []

if RUN_WINDOW_ABLATION:
    print(f"\n{'=' * 78}\nWINDOW-LENGTH ABLATION\n{'=' * 78}")
    print("  Tests whether the regime-stratified finding survives a different")
    print("  temporal resolution. If it only holds at L=24, that is a fragile")
    print("  result and must be reported as such.\n")

    for L in ABLATION_WINDOWS:
        cdir = cfg.client_dir(L)
        if not cdir.exists():
            print(f"  L={L}: client artifacts not found at {cdir}, skipping.")
            continue
        cl = load_clients(L, DEVICE)
        print(f"  L={L} (K={len(cl)}) ...")
        t0 = time.time()
        rows = []
        for seed in ABLATION_SEEDS:
            loc = train_local_only(cl, L, seed)
            fed, _ = train_federated(cl, L, seed, mu=0.0)
            rows += evaluate_all(cl, fed, loc,
                                 {"experiment": "window", "algo": "FedAvg",
                                  "mu": 0.0, "window_length": L,
                                  "local_epochs": cfg.LOCAL_EPOCHS, "seed": seed})
        df = pd.DataFrame(rows)
        agg = df.groupby(["source_name", "network_class", "regime"])["test_mae"].mean().reset_index()
        s = stratified_benefit(agg)
        s.update({"window_length": L})
        win_summaries.append(s)
        all_rows += rows
        print(f"    I2 median {s['i2_median']:+.1f}% | CESNET median {s['cesnet_median']:+.1f}% | "
              f"p={s['mannwhitney_p']:.5f} | W/T/L {s['wins']}/{s['ties']}/{s['losses']} | "
              f"{time.time() - t0:.0f}s")

    # Fold in the already-trained L=24 result for a complete picture
    try:
        pcm = pd.read_parquet(cfg.METRICS_DIR / "per_client_metrics.parquet")
        agg24 = (pcm.groupby(["source_name", "network_class", "regime"])["test_mae"]
                   .mean().reset_index())
        s24 = stratified_benefit(agg24)
        s24.update({"window_length": cfg.PRIMARY_WINDOW_LENGTH})
        win_summaries.append(s24)
    except Exception as e:
        print(f"  (could not fold in L=24 from Notebook 02: {e})")

    wa = pd.DataFrame(win_summaries).sort_values("window_length")
    wa.to_csv(cfg.METRICS_DIR / "window_ablation.csv", index=False)
    print("\n  WINDOW-LENGTH ROBUSTNESS OF THE CENTRAL CLAIM\n")
    print(wa[["window_length", "i2_median", "cesnet_median",
              "mannwhitney_p", "wins", "ties", "losses"]].round(4).to_string(index=False))
    if (wa["mannwhitney_p"] < 0.05).all():
        print("\n  -> Stratification is SIGNIFICANT at every window length tested.")
        print("     The central claim is robust to temporal resolution.")
    else:
        print("\n  -> Stratification is NOT significant at every window length.")
        print("     Report this honestly and scope the claim to where it holds.")

# %%
# =============================================================================
# CELL 5 -- Local-epoch ablation (communication vs computation)
# =============================================================================
ep_summaries = []

if RUN_LOCAL_EPOCH_ABLATION:
    L = cfg.PRIMARY_WINDOW_LENGTH
    cl = load_clients(L, DEVICE)
    print(f"\n{'=' * 78}\nLOCAL-EPOCH ABLATION (L={L})\n{'=' * 78}")
    print("  Rounds held fixed, so communication cost is constant while local")
    print("  computation scales with E. Feeds the Figure 7 crossover argument.\n")

    loc_by_seed = {s: train_local_only(cl, L, s) for s in ABLATION_SEEDS}

    for E in ABLATION_LOCAL_EPOCHS:
        print(f"  E={E} ...")
        t0 = time.time()
        rows = []
        for seed in ABLATION_SEEDS:
            fed, _ = train_federated(cl, L, seed, mu=0.0, local_epochs=E)
            rows += evaluate_all(cl, fed, loc_by_seed[seed],
                                 {"experiment": "local_epochs", "algo": "FedAvg",
                                  "mu": 0.0, "window_length": L,
                                  "local_epochs": E, "seed": seed})
        df = pd.DataFrame(rows)
        agg = df.groupby(["source_name", "network_class", "regime"])["test_mae"].mean().reset_index()
        s = stratified_benefit(agg)
        s.update({"local_epochs": E,
                  "fed_mae": df[df.regime == "Federated"]["test_mae"].mean()})
        ep_summaries.append(s)
        all_rows += rows
        print(f"    fed MAE {s['fed_mae']:.4f} | I2 median {s['i2_median']:+.1f}% | "
              f"{time.time() - t0:.0f}s")

    ea = pd.DataFrame(ep_summaries)
    ea.to_csv(cfg.METRICS_DIR / "local_epoch_ablation.csv", index=False)
    print(f"\n{ea[['local_epochs', 'fed_mae', 'i2_median', 'cesnet_median']].round(4).to_string(index=False)}")

# %%
# =============================================================================
# CELL 6 -- Save and plot
# =============================================================================
if all_rows:
    pd.DataFrame(all_rows).to_parquet(cfg.METRICS_DIR / "ablation_results.parquet",
                                      index=False)
    print(f"\n  Saved {len(all_rows):,} ablation rows.")

# --- Figure 11: FedProx vs FedAvg ---
if RUN_FEDPROX and summaries:
    fx = pd.DataFrame(summaries)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.2))
    x = np.arange(len(fx))
    a1.bar(x, fx["fed_mae"], 0.55, color=cfg.REGIME_COLORS["Federated"],
           edgecolor="black", linewidth=0.4)
    a1.set_xticks(x); a1.set_xticklabels(fx["algo"], rotation=25, ha="right", fontsize=7.5)
    a1.set_ylabel("Federated test MAE")
    a1.set_title("A: Aggregation algorithm", fontsize=10, loc="left")
    a1.grid(axis="y", linestyle=":", alpha=0.5)

    w = 0.36
    a2.bar(x - w / 2, fx["i2_median"], w, label="Internet2",
           color=cfg.NETWORK_COLORS["Internet2"], edgecolor="black", linewidth=0.4)
    a2.bar(x + w / 2, fx["cesnet_median"], w, label="CESNET3",
           color=cfg.NETWORK_COLORS["CESNET3"], edgecolor="black", linewidth=0.4)
    a2.axhline(0, color="black", linewidth=0.8)
    a2.set_xticks(x); a2.set_xticklabels(fx["algo"], rotation=25, ha="right", fontsize=7.5)
    a2.set_ylabel("Median benefit (%)")
    a2.set_title("B: Stratification persists", fontsize=10, loc="left")
    a2.legend(fontsize=8); a2.grid(axis="y", linestyle=":", alpha=0.5)

    plt.suptitle("Figure 11: FedAvg vs FedProx under extreme client heterogeneity",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    cfg.save_figure("F11_fedprox")

# --- Figure 12: window-length robustness ---
if RUN_WINDOW_ABLATION and win_summaries:
    wa = pd.DataFrame(win_summaries).sort_values("window_length")
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    x = np.arange(len(wa)); w = 0.36
    ax.bar(x - w / 2, wa["i2_median"], w, label="Internet2",
           color=cfg.NETWORK_COLORS["Internet2"], edgecolor="black", linewidth=0.4)
    ax.bar(x + w / 2, wa["cesnet_median"], w, label="CESNET3",
           color=cfg.NETWORK_COLORS["CESNET3"], edgecolor="black", linewidth=0.4)
    for i, p in enumerate(wa["mannwhitney_p"]):
        ax.text(i, max(wa["i2_median"]) * 1.06, f"p={p:.4f}",
                ha="center", fontsize=7.5, color="gray")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L={int(v)}" for v in wa["window_length"]])
    ax.set_ylabel("Median federation benefit (%)")
    ax.set_title("Figure 12: Stratification across window lengths",
                 fontsize=11, loc="left")
    ax.legend(fontsize=8); ax.grid(axis="y", linestyle=":", alpha=0.5)
    plt.tight_layout()
    cfg.save_figure("F12_window_ablation")

# --- Figure 13: local-epoch tradeoff ---
if RUN_LOCAL_EPOCH_ABLATION and ep_summaries:
    ea = pd.DataFrame(ep_summaries).sort_values("local_epochs")
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot(ea["local_epochs"], ea["fed_mae"], "o-", linewidth=2,
            color=cfg.REGIME_COLORS["Federated"])
    ax.set_xlabel("Local epochs per round (E)")
    ax.set_ylabel("Federated test MAE")
    ax.set_title("Figure 13: Computation per round vs accuracy\n"
                 "(communication cost held constant)", fontsize=10, loc="left")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    cfg.save_figure("F13_local_epochs")

print("\n" + "=" * 78)
print("NOTEBOOK 05 COMPLETE")
print("=" * 78)
