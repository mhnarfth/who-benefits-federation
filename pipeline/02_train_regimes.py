# %% [markdown]
# # Notebook 02 -- Training the Three Regimes
#
# **INDIS 2026:** *Learning Without Centralizing*
#
# Trains three regimes at **equal total epoch budget** (500 = 100 rounds x 5
# local epochs, derived in `fed_config`):
#
# | Regime | What it is | Role in the paper |
# |---|---|---|
# | Centralized | All client windows pooled, one model | Privacy-violating **upper bound** |
# | Local-only | 20 independent models, no communication | Status-quo **lower bound** |
# | Federated | FedAvg over 20 clients | **The proposal** |
#
# **Performance note:** the entire federation is ~2 MB of float32. Everything
# is loaded to the GPU once and never leaves. No DataLoader, no host-device
# transfer in the training loop -- with models this small that overhead would
# dominate real compute.
#
# **Model selection asymmetry (state this in the paper):** centralized and
# local-only use validation early stopping. Standard FedAvg does *not*
# early-stop locally; instead the global model is selected by best mean
# validation loss across rounds.
#
# **Outputs:** `models/`, `metrics/histories.parquet`,
# `metrics/comm_costs.json`, `metrics/notebook02_manifest.json`

# %%
# =============================================================================
# CELL 0 -- Imports, config, device
# =============================================================================
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
import fed_config as cfg

cfg.ensure_dirs()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 72)
print("NOTEBOOK 02 -- TRAINING THE THREE REGIMES")
print("=" * 72)
print(f"  Device            : {DEVICE}")
if DEVICE.type == "cuda":
    print(f"  GPU               : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM              : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  Window length     : {cfg.PRIMARY_WINDOW_LENGTH}")
print(f"  Fed rounds        : {cfg.FED_ROUNDS} x {cfg.LOCAL_EPOCHS} local epochs")
print(f"  Equal budget      : {cfg.TOTAL_EPOCH_BUDGET} epochs for every regime")
print(f"  Seeds             : {cfg.SEEDS}")
print("=" * 72)

# %%
# =============================================================================
# CELL 1 -- Model definition
# =============================================================================

class MLPAutoencoder(nn.Module):
    """Symmetric MLP autoencoder: L -> 128 -> 64 -> d -> 64 -> 128 -> L.

    Sizing is bounded by DATA per client (~1,000 windows), not by GPU
    capacity. At ~24K parameters this is roughly 24 params/sample -- near the
    ceiling before local overfitting destabilizes weight averaging. The
    latent-dim ablation demonstrates this rather than asserting it.

    No BatchNorm: running statistics are stateful buffers that FedAvg would
    have to average across non-IID clients, which is a known failure mode and
    an unnecessary complication here.
    """

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


def count_params(model):
    return sum(p.numel() for p in model.parameters())


_probe = MLPAutoencoder(cfg.PRIMARY_WINDOW_LENGTH)
N_PARAMS = count_params(_probe)
MODEL_BYTES = N_PARAMS * 4
print(f"  Parameters        : {N_PARAMS:,}")
print(f"  Model size        : {MODEL_BYTES / 1024:.1f} KB (float32)")
del _probe

# %%
# =============================================================================
# CELL 2 -- Load all clients to GPU (once)
# =============================================================================

def load_clients(window_length, device):
    """Load every training client's windows straight onto the device."""
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


L = cfg.PRIMARY_WINDOW_LENGTH
clients = load_clients(L, DEVICE)
K = len(clients)

n_k = {cid: c["train"].shape[0] for cid, c in clients.items()}
n_total = sum(n_k.values())
total_mb = sum(c[s].numel() * 4 for c in clients.values()
               for s in ["train", "val", "test"]) / 1e6

print(f"\n  Clients loaded    : K = {K}")
print(f"  Total train windows: {n_total:,}")
print(f"  Resident on device: {total_mb:.2f} MB")
for cid, c in clients.items():
    print(f"    [{cid:02d}] {c['network_class']:10s} {c['source_name']:14s} "
          f"train={c['train'].shape[0]:5d}  val={c['val'].shape[0]:4d}  test={c['test'].shape[0]:4d}")

# %%
# =============================================================================
# CELL 3 -- Training primitives
# =============================================================================

@torch.no_grad()
def eval_mse(model, X):
    """Mean per-window reconstruction MSE. Empty tensor -> nan."""
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


def run_epochs(model, X, n_epochs, gen, lr=None, wd=None, batch_size=None, opt=None):
    """Run `n_epochs` of Adam over X. Returns (model, optimizer, mean loss)."""
    lr = lr if lr is not None else cfg.LEARNING_RATE
    wd = wd if wd is not None else cfg.WEIGHT_DECAY
    batch_size = batch_size or cfg.BATCH_SIZE
    if opt is None:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    n = X.shape[0]
    model.train()
    last = float("nan")
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=X.device, generator=gen)
        running, nb = 0.0, 0
        for i in range(0, n, batch_size):
            xb = X[perm[i:i + batch_size]]
            opt.zero_grad(set_to_none=True)
            recon, _ = model(xb)
            loss = F.mse_loss(recon, xb)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        last = running / max(nb, 1)
    return model, opt, last


def make_generator(seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

# %%
# =============================================================================
# CELL 4 -- Regime 1: Centralized (privacy-violating upper bound)
# =============================================================================

def train_centralized(clients, input_dim, seed, verbose=True):
    cfg.set_seed(seed)
    gen = make_generator(seed, DEVICE)

    X_tr = torch.cat([c["train"] for c in clients.values()], dim=0)
    X_va = torch.cat([c["val"] for c in clients.values()], dim=0)

    model = MLPAutoencoder(input_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                           weight_decay=cfg.WEIGHT_DECAY)

    best_val, best_state, best_epoch, patience = float("inf"), None, -1, 0
    history = []

    for ep in range(cfg.CENTRALIZED_MAX_EPOCHS):
        model, opt, tr_loss = run_epochs(model, X_tr, 1, gen, opt=opt)
        va_loss = eval_mse(model, X_va)
        history.append({"regime": "Centralized", "seed": seed, "step": ep,
                        "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val - 1e-9:
            best_val, best_epoch, patience = va_loss, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                if verbose:
                    print(f"    early stop @ epoch {ep} (best {best_epoch}, val {best_val:.5f})")
                break

    model.load_state_dict(best_state)
    return model, history, {"best_epoch": best_epoch, "best_val": best_val}

# %%
# =============================================================================
# CELL 5 -- Regime 2: Local-only (status-quo lower bound)
# =============================================================================

def train_local_only(clients, input_dim, seed, verbose=True):
    models, history, info = {}, [], {}

    for cid, c in clients.items():
        cfg.set_seed(seed + cid)          # distinct init per client
        gen = make_generator(seed + cid, DEVICE)

        model = MLPAutoencoder(input_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                               weight_decay=cfg.WEIGHT_DECAY)

        best_val, best_state, best_epoch, patience = float("inf"), None, -1, 0
        for ep in range(cfg.LOCAL_ONLY_MAX_EPOCHS):
            model, opt, tr_loss = run_epochs(model, c["train"], 1, gen, opt=opt)
            va_loss = eval_mse(model, c["val"])
            history.append({"regime": "Local-only", "seed": seed, "client_id": cid,
                            "step": ep, "train_loss": tr_loss, "val_loss": va_loss})
            if va_loss < best_val - 1e-9:
                best_val, best_epoch, patience = va_loss, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience += 1
                if patience >= cfg.EARLY_STOP_PATIENCE:
                    break

        model.load_state_dict(best_state)
        models[cid] = model
        info[cid] = {"best_epoch": best_epoch, "best_val": best_val}
        if verbose:
            print(f"    [{cid:02d}] {c['source_name']:14s} best val {best_val:.5f} @ {best_epoch}")

    return models, history, info

# %%
# =============================================================================
# CELL 6 -- Regime 3: FedAvg
# =============================================================================

def train_fedavg(clients, input_dim, seed, verbose=True):
    """Standard McMahan FedAvg with full participation.

    Aggregation is weighted by each client's training-set size:
        theta^{t+1} = sum_k (n_k / n) * theta_k^{t+1}

    No local early stopping (not part of FedAvg). The returned global model is
    the round whose mean validation loss across clients was lowest.
    """
    cfg.set_seed(seed)
    global_model = MLPAutoencoder(input_dim).to(DEVICE)

    weights = {cid: n_k[cid] / n_total for cid in clients}
    best_val, best_state, best_round = float("inf"), None, -1
    history = []

    for rnd in range(cfg.FED_ROUNDS):
        global_state = copy.deepcopy(global_model.state_dict())
        agg = {k: torch.zeros_like(v, dtype=torch.float32)
               for k, v in global_state.items()}

        round_train = []
        for cid, c in clients.items():
            gen = make_generator(seed * 10_000 + rnd * 100 + cid, DEVICE)
            local = MLPAutoencoder(input_dim).to(DEVICE)
            local.load_state_dict(global_state)
            local, _, tr_loss = run_epochs(local, c["train"], cfg.LOCAL_EPOCHS, gen)
            round_train.append(tr_loss)

            w = weights[cid]
            for k, v in local.state_dict().items():
                agg[k] += w * v.float()

        global_model.load_state_dict(agg)

        val_per_client = {cid: eval_mse(global_model, c["val"])
                          for cid, c in clients.items()}
        mean_val = float(np.nanmean(list(val_per_client.values())))
        mean_train = float(np.mean(round_train))

        history.append({"regime": "Federated", "seed": seed, "step": rnd,
                        "train_loss": mean_train, "val_loss": mean_val})

        if mean_val < best_val - 1e-9:
            best_val, best_round = mean_val, rnd
            best_state = copy.deepcopy(global_model.state_dict())

        if verbose and (rnd % 10 == 0 or rnd == cfg.FED_ROUNDS - 1):
            print(f"    round {rnd:3d}  train {mean_train:.5f}  val {mean_val:.5f}")

    global_model.load_state_dict(best_state)
    if verbose:
        print(f"    best round {best_round} (mean val {best_val:.5f})")
    return global_model, history, {"best_round": best_round, "best_val": best_val,
                                   "rounds_run": cfg.FED_ROUNDS}

# %%
# =============================================================================
# CELL 7 -- Multi-seed execution
# =============================================================================
all_history, run_info = [], {}
t_start = time.time()

for seed in cfg.SEEDS:
    print(f"\n{'=' * 72}\nSEED {seed}\n{'=' * 72}")
    seed_dir = cfg.MODELS_DIR / f"L{L}" / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    print("  [1/3] Centralized ...")
    t0 = time.time()
    m_cen, h_cen, i_cen = train_centralized(clients, L, seed)
    torch.save(m_cen.state_dict(), seed_dir / "centralized.pt")
    print(f"        done in {time.time() - t0:.1f}s")

    print("  [2/3] Local-only ...")
    t0 = time.time()
    m_loc, h_loc, i_loc = train_local_only(clients, L, seed, verbose=False)
    for cid, m in m_loc.items():
        torch.save(m.state_dict(), seed_dir / f"local_{cid:02d}.pt")
    print(f"        done in {time.time() - t0:.1f}s ({len(m_loc)} models)")

    print("  [3/3] Federated (FedAvg) ...")
    t0 = time.time()
    m_fed, h_fed, i_fed = train_fedavg(clients, L, seed)
    torch.save(m_fed.state_dict(), seed_dir / "federated.pt")
    print(f"        done in {time.time() - t0:.1f}s")

    all_history += h_cen + h_loc + h_fed
    run_info[str(seed)] = {"centralized": i_cen, "federated": i_fed,
                           "local_only": {str(k): v for k, v in i_loc.items()}}

hist_df = pd.DataFrame(all_history)
hist_path = cfg.METRICS_DIR / "histories.parquet"
hist_df.to_parquet(hist_path, index=False)

print(f"\n  Total wall time  : {time.time() - t_start:.1f}s")
print(f"  History rows     : {len(hist_df):,}  -> {hist_path.name}")

# %%
# =============================================================================
# CELL 8 -- Communication accounting and crossover analysis
# =============================================================================
# HONEST FRAMING: at this data scale FedAvg costs MORE bytes than shipping the
# hourly series. Federation's value here is the legal/administrative boundary
# (raw telemetry never crosses an NDA line), not bandwidth. But FedAvg's cost
# is FLAT in data volume while centralization grows LINEARLY, so a crossover
# exists -- and real deployments (raw flows, sub-hourly, multivariate) sit well
# past it. We compute that crossover rather than asserting a savings claim.

BYTES_PER_POINT_FLOAT64 = 8
mean_rounds = float(np.mean([run_info[str(s)]["federated"]["rounds_run"] for s in cfg.SEEDS]))

fed_bytes = mean_rounds * K * 2 * MODEL_BYTES          # down + up, every round
hourly_points_per_client = float(np.mean([c["train"].shape[0] + c["val"].shape[0]
                                          + c["test"].shape[0] for c in clients.values()]))
central_hourly_bytes = K * hourly_points_per_client * BYTES_PER_POINT_FLOAT64

# Crossover: per-client data volume at which centralization cost == FedAvg cost
crossover_bytes_per_client = mean_rounds * 2 * MODEL_BYTES

scenarios = [
    ("Hourly univariate (this paper)", hourly_points_per_client * BYTES_PER_POINT_FLOAT64),
    ("5-min univariate",               hourly_points_per_client * 12 * BYTES_PER_POINT_FLOAT64),
    ("5-min, 12-metric multivariate",  hourly_points_per_client * 12 * 12 * BYTES_PER_POINT_FLOAT64),
    ("1-sec, 12-metric multivariate",  hourly_points_per_client * 3600 * 12 * BYTES_PER_POINT_FLOAT64),
    ("Raw NetFlow records (~1.5 GB)",  1.5e9),
]

comm = {
    "n_params": N_PARAMS,
    "model_bytes": MODEL_BYTES,
    "n_clients": K,
    "mean_rounds": mean_rounds,
    "fedavg_total_bytes": fed_bytes,
    "centralize_hourly_total_bytes": central_hourly_bytes,
    "fedavg_vs_hourly_ratio": fed_bytes / central_hourly_bytes,
    "crossover_bytes_per_client": crossover_bytes_per_client,
    "scenarios": [
        {"label": lbl, "bytes_per_client": b,
         "centralize_total_bytes": K * b,
         "cheaper": "Federate" if K * b > fed_bytes else "Centralize"}
        for lbl, b in scenarios
    ],
}
with open(cfg.METRICS_DIR / "comm_costs.json", "w") as f:
    json.dump(comm, f, indent=2)

print("\n" + "=" * 72)
print("COMMUNICATION ACCOUNTING")
print("=" * 72)
print(f"  Model size                 : {MODEL_BYTES / 1024:.1f} KB")
print(f"  FedAvg total ({mean_rounds:.0f} rounds) : {fed_bytes / 1e6:.1f} MB")
print(f"  Centralize hourly series   : {central_hourly_bytes / 1e3:.1f} KB")
print(f"  Ratio                      : FedAvg costs {fed_bytes / central_hourly_bytes:.0f}x MORE")
print(f"\n  Crossover per client       : {crossover_bytes_per_client / 1e6:.1f} MB")
print(f"  (FedAvg is flat in data volume; centralization grows linearly)\n")
print(f"  {'Scenario':<34s} {'Bytes/client':>14s}  {'Cheaper':>10s}")
print("  " + "-" * 62)
for s in comm["scenarios"]:
    print(f"  {s['label']:<34s} {s['bytes_per_client']:>14,.0f}  {s['cheaper']:>10s}")
print("=" * 72)

# %%
# =============================================================================
# CELL 9 -- Manifest
# =============================================================================
manifest = {
    "notebook": "02_train_regimes",
    "config_signature": cfg.config_signature(),
    "device": str(DEVICE),
    "window_length": L,
    "n_clients": K,
    "n_params": N_PARAMS,
    "model_bytes": MODEL_BYTES,
    "seeds": cfg.SEEDS,
    "run_info": run_info,
    "artifacts": {
        "models_dir": str(cfg.MODELS_DIR / f"L{L}"),
        "histories": str(hist_path),
        "comm_costs": str(cfg.METRICS_DIR / "comm_costs.json"),
    },
}
mpath = cfg.METRICS_DIR / "notebook02_manifest.json"
with open(mpath, "w") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 72)
print("NOTEBOOK 02 COMPLETE")
print("=" * 72)
print(f"  Regimes trained : {cfg.REGIMES}")
print(f"  Seeds           : {len(cfg.SEEDS)}")
print(f"  Models          : {cfg.MODELS_DIR / f'L{L}'}")
print(f"  Manifest        : {mpath.name}")
print("\n  Next: 03_analysis_figures.ipynb")
print("=" * 72)
