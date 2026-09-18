# %% [markdown]
# # Notebook 03 -- Analysis and Figures
#
# **INDIS 2026:** *Learning Without Centralizing*
#
# Produces Figures 3-8 and Tables 3-4 from the artifacts written by
# Notebooks 01 and 02. No training happens here.
#
# **Caching discipline:** embeddings are computed and cached to `.npz` BEFORE
# any UMAP call. UMAP is the slowest and most re-run-prone step; you should
# never recompute embeddings just to change a plot colour.
#
# **Outputs:**
# - `charts/F3_convergence`, `F4_per_client`, `F5_embeddings`,
#   `F6_probe_confusion`, `F7_comm_crossover`, `F8_isp_transfer`
# - `metrics/table3_performance.csv`, `metrics/table4_communication.csv`
# - `metrics/embeddings_*.npz`, `metrics/per_client_metrics.parquet`

# %%
# =============================================================================
# CELL 0 -- Imports and artifact loading
# =============================================================================
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

# Reuse the model class from Notebook 02 rather than redefining it.
_nb02 = Path(cfg.PROJECT_ROOT) / "notebooks" / "02_train_regimes.py"
if _nb02.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_nb02mod", _nb02)
    # NOTE: executing 02 as a module would retrain. Define the class locally
    # instead -- kept byte-identical to Notebook 02's definition.
import torch.nn as nn


class MLPAutoencoder(nn.Module):
    """Must stay byte-identical to the definition in Notebook 02."""

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


cfg.ensure_dirs()
cfg.setup_matplotlib()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L = cfg.PRIMARY_WINDOW_LENGTH

with open(cfg.METRICS_DIR / "notebook02_manifest.json") as f:
    nb02 = json.load(f)
with open(cfg.METRICS_DIR / "comm_costs.json") as f:
    comm = json.load(f)
hist = pd.read_parquet(cfg.METRICS_DIR / "histories.parquet")

print("=" * 72)
print("NOTEBOOK 03 -- ANALYSIS AND FIGURES")
print("=" * 72)
print(f"  Device        : {DEVICE}")
print(f"  Window length : {L}")
print(f"  Seeds         : {nb02['seeds']}")
print(f"  History rows  : {len(hist):,}")
print("=" * 72)

# %%
# =============================================================================
# CELL 1 -- Reload clients and models
# =============================================================================

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


def load_isp_probe(window_length, device):
    fp = cfg.client_dir(window_length) / "probe_isp.npz"
    if not fp.exists():
        return None
    d = np.load(fp, ispw_pickle=True)
    viable = bool(d["probe_viable"]) if "probe_viable" in d else (len(d["probe"]) >= 10)
    if not viable:
        print(f"  ISP probe at L={window_length} is NOT viable "
              f"({len(d['probe'])} windows). Figure 8 will be skipped.")
        return None
    return {"probe": torch.tensor(d["probe"], dtype=torch.float32, device=device),
            "n_windows": int(len(d["probe"]))}


clients = load_clients(L, DEVICE)
K = len(clients)
isp = load_isp_probe(L, DEVICE)


def load_models(seed):
    sdir = cfg.MODELS_DIR / f"L{L}" / f"seed{seed}"
    out = {}
    m = MLPAutoencoder(L).to(DEVICE)
    m.load_state_dict(torch.load(sdir / "centralized.pt", map_location=DEVICE))
    out["Centralized"] = m.eval()
    m = MLPAutoencoder(L).to(DEVICE)
    m.load_state_dict(torch.load(sdir / "federated.pt", map_location=DEVICE))
    out["Federated"] = m.eval()
    loc = {}
    for cid in clients:
        mm = MLPAutoencoder(L).to(DEVICE)
        mm.load_state_dict(torch.load(sdir / f"local_{cid:02d}.pt", map_location=DEVICE))
        loc[cid] = mm.eval()
    out["Local-only"] = loc
    return out


print(f"  Clients       : {K}")
print(f"  ISP probe    : {'viable, ' + str(isp['n_windows']) + ' windows' if isp else 'unavailable'}")

# %%
# =============================================================================
# CELL 2 -- Per-client test metrics (all regimes, all seeds)
# =============================================================================

@torch.no_grad()
def recon_metrics(model, X):
    if X.shape[0] == 0:
        return float("nan"), float("nan")
    recon, _ = model(X)
    return (float(F.mse_loss(recon, X).item()),
            float(torch.mean(torch.abs(recon - X)).item()))


rows = []
for seed in nb02["seeds"]:
    models = load_models(seed)
    for cid, c in clients.items():
        for regime in cfg.REGIMES:
            m = models[regime][cid] if regime == "Local-only" else models[regime]
            mse, mae = recon_metrics(m, c["test"])
            rows.append({"seed": seed, "client_id": cid,
                         "network_class": c["network_class"],
                         "source_name": c["source_name"], "regime": regime,
                         "test_mse": mse, "test_mae": mae,
                         "n_train": c["train"].shape[0]})

pcm = pd.DataFrame(rows)
pcm.to_parquet(cfg.METRICS_DIR / "per_client_metrics.parquet", index=False)
print(f"\n  Per-client metrics: {len(pcm):,} rows "
      f"({len(nb02['seeds'])} seeds x {K} clients x {len(cfg.REGIMES)} regimes)")

# %%
# =============================================================================
# CELL 3 -- Figure 3: Convergence curves
# =============================================================================
fig, ax = plt.subplots(figsize=(5.2, 3.6))

# Centralized and Federated: mean +/- std across seeds
for regime in ["Centralized", "Federated"]:
    h = hist[hist["regime"] == regime]
    g = h.groupby("step")["val_loss"].agg(["mean", "std"]).reset_index()
    ax.plot(g["step"], g["mean"], color=cfg.REGIME_COLORS[regime],
            linewidth=2, label=regime)
    ax.fill_between(g["step"], g["mean"] - g["std"].fillna(0),
                    g["mean"] + g["std"].fillna(0),
                    color=cfg.REGIME_COLORS[regime], alpha=0.18)

# Local-only: mean across clients+seeds, with min-max band across clients.
# The BAND WIDTH is the story -- some silos do fine alone, others do not.
hl = hist[hist["regime"] == "Local-only"]
if len(hl):
    per_step = hl.groupby(["step", "client_id"])["val_loss"].mean().reset_index()
    g = per_step.groupby("step")["val_loss"].agg(["mean", "min", "max"]).reset_index()
    ax.plot(g["step"], g["mean"], color=cfg.REGIME_COLORS["Local-only"],
            linewidth=2, linestyle="--", label="Local-only (mean)")
    ax.fill_between(g["step"], g["min"], g["max"],
                    color=cfg.REGIME_COLORS["Local-only"], alpha=0.15,
                    label="Local-only (client min-max)")

ax.set_yscale("log")
ax.set_xlabel("Training step (epoch for baselines, round for FedAvg)")
ax.set_ylabel("Validation reconstruction MSE")
ax.set_title("Figure 3: Convergence by training regime", fontsize=11, loc="left")
ax.grid(True, linestyle=":", alpha=0.5)
ax.legend(fontsize=8)
plt.tight_layout()
cfg.save_figure("F3_convergence")

# %%
# =============================================================================
# CELL 4 -- Figure 4: Per-client reconstruction (seed-averaged)
# =============================================================================
agg = (pcm.groupby(["client_id", "source_name", "network_class", "regime"])
          ["test_mae"].agg(["mean", "std"]).reset_index())
order = sorted(clients.keys(), key=lambda c: (clients[c]["network_class"],
                                              clients[c]["source_name"]))
labels = [clients[c]["source_name"] for c in order]
x = np.arange(len(order))
width = 0.26

fig, ax = plt.subplots(figsize=(7.2, 3.4))
for i, regime in enumerate(cfg.REGIMES):
    sub = agg[agg["regime"] == regime].set_index("client_id")
    means = [sub.loc[c, "mean"] if c in sub.index else np.nan for c in order]
    errs = [sub.loc[c, "std"] if c in sub.index else 0 for c in order]
    ax.bar(x + (i - 1) * width, means, width, yerr=errs, capsize=2,
           label=regime, color=cfg.REGIME_COLORS[regime], edgecolor="black",
           linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
ax.set_ylabel("Test reconstruction MAE")
ax.set_title("Figure 4: Per-client reconstruction error "
             "(mean +/- std over seeds)", fontsize=11, loc="left")
ax.grid(axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=8)

n_i2 = sum(1 for c in order if clients[c]["network_class"] == "Internet2")
ax.axvline(n_i2 - 0.5, color="gray", linestyle="-", linewidth=1, alpha=0.6)
ax.text(n_i2 / 2, ax.get_ylim()[1] * 0.95, "Internet2", ha="center", fontsize=8, color="gray")
ax.text(n_i2 + (len(order) - n_i2) / 2, ax.get_ylim()[1] * 0.95, "CESNET3",
        ha="center", fontsize=8, color="gray")
plt.tight_layout()
cfg.save_figure("F4_per_client")

# %%
# =============================================================================
# CELL 5 -- Compute and CACHE embeddings (before any UMAP)
# =============================================================================
PRIMARY_SEED = cfg.PRIMARY_SEED
models = load_models(PRIMARY_SEED)


@torch.no_grad()
def embed(model, X):
    if X.shape[0] == 0:
        return np.empty((0, cfg.LATENT_DIM), dtype=np.float32)
    _, z = model(X)
    return z.cpu().numpy()


emb_store = {}
for regime in ["Centralized", "Federated"]:
    Z, y_net, y_cid = [], [], []
    for cid, c in clients.items():
        z = embed(models[regime], c["test"])
        Z.append(z)
        y_net += [c["network_class"]] * len(z)
        y_cid += [cid] * len(z)
    emb_store[regime] = {"Z": np.concatenate(Z), "net": np.array(y_net),
                         "cid": np.array(y_cid)}

# Local-only: each client embedded by its OWN model. Latent axes are not
# comparable across independently-trained models -- this is exactly why
# local-only produces no shared representation space, which is the point.
Z, y_net, y_cid = [], [], []
for cid, c in clients.items():
    z = embed(models["Local-only"][cid], c["test"])
    Z.append(z)
    y_net += [c["network_class"]] * len(z)
    y_cid += [cid] * len(z)
emb_store["Local-only"] = {"Z": np.concatenate(Z), "net": np.array(y_net),
                           "cid": np.array(y_cid)}

if isp is not None:
    for regime in ["Centralized", "Federated"]:
        emb_store[regime]["Z_isp"] = embed(models[regime], isp["probe"])

for regime, d in emb_store.items():
    fp = cfg.METRICS_DIR / f"embeddings_{regime.lower().replace('-', '_')}.npz"
    np.savez_compressed(fp, **{k: v for k, v in d.items()})
    print(f"  cached: {fp.name}  Z={d['Z'].shape}")

# %%
# =============================================================================
# CELL 6 -- Figure 5: Latent embedding projections
# =============================================================================
try:
    import umap
    def project(Z, seed=42):
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed).fit_transform(Z)
    PROJ = "UMAP"
except ImportError:
    from sklearn.manifold import TSNE
    def project(Z, seed=42):
        return TSNE(n_components=2, random_state=seed, init="pca",
                    perplexity=30).fit_transform(Z)
    PROJ = "t-SNE"
    print("  umap-learn not found -- falling back to t-SNE.")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=False, sharey=False)
for ax, regime in zip(axes, ["Centralized", "Federated"]):
    d = emb_store[regime]
    P = project(d["Z"])
    for net in ["Internet2", "CESNET3"]:
        m = d["net"] == net
        ax.scatter(P[m, 0], P[m, 1], s=3, alpha=0.45,
                   c=cfg.NETWORK_COLORS[net], label=net, linewidths=0)
    ax.set_title(regime, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].legend(fontsize=8, markerscale=3, loc="best")
plt.suptitle(f"Figure 5: Latent space ({PROJ} projection, seed {PRIMARY_SEED})",
             fontsize=11, y=1.01)
plt.tight_layout()
cfg.save_figure("F5_embeddings")

# %%
# =============================================================================
# CELL 7 -- Figure 6 + probe accuracy: linear probe on frozen embeddings
# =============================================================================
probe_results = {}
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7))

for ax, regime in zip(axes, cfg.REGIMES):
    d = emb_store[regime]
    Z, y = d["Z"], d["net"]
    rng = np.random.RandomState(cfg.PRIMARY_SEED)
    idx = rng.permutation(len(Z))
    split = int(0.7 * len(Z))
    tr, te = idx[:split], idx[split:]

    sc = StandardScaler().fit(Z[tr])
    clf = LogisticRegression(max_iter=2000, random_state=cfg.PRIMARY_SEED)
    clf.fit(sc.transform(Z[tr]), y[tr])
    pred = clf.predict(sc.transform(Z[te]))
    acc = accuracy_score(y[te], pred)
    probe_results[regime] = acc

    cm = confusion_matrix(y[te], pred, labels=["Internet2", "CESNET3"], normalize="true")
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["I2", "CES"], fontsize=8)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["I2", "CES"], fontsize=8)
    ax.set_title(f"{regime}\nacc = {acc:.3f}", fontsize=9)

plt.suptitle("Figure 6: Network-class probe on frozen embeddings",
             fontsize=11, y=1.03)
plt.tight_layout()
cfg.save_figure("F6_probe_confusion")
print("  Probe accuracy:", {k: round(v, 4) for k, v in probe_results.items()})

# %%
# =============================================================================
# CELL 8 -- Figure 7: Communication crossover (HONEST framing)
# =============================================================================
# FedAvg cost is FLAT in data volume; centralization grows LINEARLY. At this
# paper's scale centralization is cheaper in bytes -- we say so. The crossover
# is what matters operationally, and real deployments sit past it.
fed_total = comm["fedavg_total_bytes"]
Kc = comm["n_clients"]

bpc = np.logspace(3, 10, 200)                # bytes per client
central_line = Kc * bpc
fed_line = np.full_like(bpc, fed_total)
crossover = comm["crossover_bytes_per_client"]

fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.loglog(bpc, central_line, color="#4C72B0", linewidth=2,
          label="Centralize raw data (grows linearly)")
ax.loglog(bpc, fed_line, color="#D45087", linewidth=2,
          label=f"FedAvg weights ({comm['mean_rounds']:.0f} rounds, flat)")
ax.axvline(crossover, color="gray", linestyle="--", linewidth=1.2)
ax.text(crossover * 1.15, fed_total * 3, "crossover", fontsize=8,
        rotation=90, color="gray")

for s in comm["scenarios"]:
    ax.plot(s["bytes_per_client"], Kc * s["bytes_per_client"], "o",
            color="#2F4B7C", markersize=5, zorder=5)
    ax.annotate(s["label"], (s["bytes_per_client"], Kc * s["bytes_per_client"]),
                textcoords="offset points", xytext=(6, -3), fontsize=6.5)

ax.set_xlabel("Data volume per client (bytes)")
ax.set_ylabel("Total bytes transferred")
ax.set_title("Figure 7: Communication crossover", fontsize=11, loc="left")
ax.grid(True, which="both", linestyle=":", alpha=0.4)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
cfg.save_figure("F7_comm_crossover")

# %%
# =============================================================================
# CELL 9 -- Figure 8: ISP unseen-network transfer
# =============================================================================
if isp is not None:
    d = emb_store["Federated"]
    Z_all = np.vstack([d["Z"], d["Z_isp"]])
    P = project(Z_all)
    n_train = len(d["Z"])

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for net in ["Internet2", "CESNET3"]:
        m = d["net"] == net
        ax.scatter(P[:n_train][m, 0], P[:n_train][m, 1], s=4, alpha=0.35,
                   c=cfg.NETWORK_COLORS[net], label=net, linewidths=0)
    ax.scatter(P[n_train:, 0], P[n_train:, 1], s=70, marker="*",
               c=cfg.NETWORK_COLORS["ISP"], edgecolors="black", linewidths=0.5,
               label=f"ISP (unseen, n={isp['n_windows']})", zorder=5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Figure 8: Unseen-network transfer\n"
                 "(federated encoder, ISP never in training)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, markerscale=1.5)
    plt.tight_layout()
    cfg.save_figure("F8_isp_transfer")
else:
    print("  Figure 8 SKIPPED -- ISP probe not viable at this window length.")

# %%
# =============================================================================
# CELL 10 -- Tables 3 and 4
# =============================================================================
t3 = (pcm.groupby("regime")[["test_mse", "test_mae"]]
         .agg(["mean", "std"]).reset_index())
t3.columns = ["regime", "mse_mean", "mse_std", "mae_mean", "mae_std"]
t3["probe_accuracy"] = t3["regime"].map(probe_results)

cen_mse = float(t3.loc[t3["regime"] == "Centralized", "mse_mean"].iloc[0])
t3["pct_of_centralized"] = (cen_mse / t3["mse_mean"] * 100).round(1)
t3.to_csv(cfg.METRICS_DIR / "table3_performance.csv", index=False)

t4 = pd.DataFrame([
    {"Regime": "Centralized", "Total bytes moved": comm["centralize_hourly_total_bytes"],
     "Raw data exposed": "Yes", "Crosses admin boundary": "Yes"},
    {"Regime": "Federated", "Total bytes moved": comm["fedavg_total_bytes"],
     "Raw data exposed": "No", "Crosses admin boundary": "No"},
    {"Regime": "Local-only", "Total bytes moved": 0,
     "Raw data exposed": "No", "Crosses admin boundary": "No"},
])
t4.to_csv(cfg.METRICS_DIR / "table4_communication.csv", index=False)

print("\nTABLE 3 -- Performance by regime\n")
print(t3.to_string(index=False))
print("\nTABLE 4 -- Communication and privacy\n")
print(t4.to_string(index=False))

print("\n" + "=" * 72)
print("NOTEBOOK 03 COMPLETE")
print("=" * 72)
print(f"  Figures -> {cfg.CHARTS_DIR}")
print(f"  Tables  -> {cfg.METRICS_DIR}")
print("=" * 72)
