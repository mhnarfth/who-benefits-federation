"""
convergence_check.py -- Confirms FedAvg has converged by 100 rounds.

WHY: across the five seeds, the best global round landed at 97, 95, 82, 98, 87
out of 100. Three of five sit in the last five rounds, which LOOKS like the
run was truncated early. The validation trajectory says otherwise (flat to the
fifth decimal over the last ten rounds), but a reviewer seeing "best round
97/100" will ask. This script answers it: run one seed to 200 rounds and
report how much the validation loss actually moves past round 100.

DOES NOT touch any existing artifact. Reads client .npz files, writes only
metrics/convergence_check.json. Nothing else is retrained or overwritten.

Run from the project root:  python convergence_check.py
Runtime: roughly 4 minutes on the H200.
"""

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import fed_config as cfg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = cfg.PRIMARY_SEED       # 42
EXTENDED_ROUNDS = 200         # double the main experiment's 100
CHECKPOINTS = [25, 50, 75, 100, 125, 150, 175, 200]


class MLPAutoencoder(nn.Module):
    """Byte-identical to the definition in Notebook 02."""

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
        clients[int(d["client_id"])] = {
            "train": torch.tensor(d["train"], dtype=torch.float32, device=device),
            "val": torch.tensor(d["val"], dtype=torch.float32, device=device),
        }
    return clients


@torch.no_grad()
def eval_mse(model, X):
    if X.shape[0] == 0:
        return float("nan")
    model.eval()
    recon, _ = model(X)
    return float(F.mse_loss(recon, X).item())


def run_epochs(model, X, n_epochs, gen):
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE,
                           weight_decay=cfg.WEIGHT_DECAY)
    n = X.shape[0]
    model.train()
    for _ in range(n_epochs):
        perm = torch.randperm(n, device=X.device, generator=gen)
        for i in range(0, n, cfg.BATCH_SIZE):
            xb = X[perm[i:i + cfg.BATCH_SIZE]]
            opt.zero_grad(set_to_none=True)
            recon, _ = model(xb)
            F.mse_loss(recon, xb).backward()
            opt.step()
    return model


def main():
    L = cfg.PRIMARY_WINDOW_LENGTH
    clients = load_clients(L, DEVICE)
    K = len(clients)
    n_k = {cid: c["train"].shape[0] for cid, c in clients.items()}
    n_total = sum(n_k.values())

    print("=" * 70)
    print("FEDAVG CONVERGENCE CHECK")
    print("=" * 70)
    print(f"  Device        : {DEVICE}")
    print(f"  Clients       : K = {K}")
    print(f"  Seed          : {SEED}")
    print(f"  Rounds        : {EXTENDED_ROUNDS} (main experiment used {cfg.FED_ROUNDS})")
    print(f"  Local epochs  : {cfg.LOCAL_EPOCHS}")
    print("=" * 70 + "\n")

    cfg.set_seed(SEED)
    global_model = MLPAutoencoder(L).to(DEVICE)
    weights = {cid: n_k[cid] / n_total for cid in clients}

    history = {}
    best_val, best_round = float("inf"), -1
    t0 = time.time()

    for rnd in range(EXTENDED_ROUNDS):
        global_state = copy.deepcopy(global_model.state_dict())
        agg = {k: torch.zeros_like(v, dtype=torch.float32)
               for k, v in global_state.items()}

        for cid, c in clients.items():
            gen = torch.Generator(device=DEVICE)
            gen.manual_seed(SEED * 10_000 + rnd * 100 + cid)
            local = MLPAutoencoder(L).to(DEVICE)
            local.load_state_dict(global_state)
            run_epochs(local, c["train"], cfg.LOCAL_EPOCHS, gen)
            w = weights[cid]
            for k, v in local.state_dict().items():
                agg[k] += w * v.float()

        global_model.load_state_dict(agg)
        mean_val = float(np.nanmean([eval_mse(global_model, c["val"])
                                     for c in clients.values()]))

        if mean_val < best_val - 1e-9:
            best_val, best_round = mean_val, rnd

        r1 = rnd + 1
        if r1 in CHECKPOINTS:
            history[r1] = mean_val
            print(f"  round {r1:3d}  mean val MSE = {mean_val:.6f}")

    elapsed = time.time() - t0

    v100 = history.get(100)
    v200 = history.get(200)
    pct = 100 * (v100 - v200) / v100 if (v100 and v200) else float("nan")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Val MSE at round 100      : {v100:.6f}")
    print(f"  Val MSE at round 200      : {v200:.6f}")
    print(f"  Improvement 100 -> 200    : {pct:.3f}%")
    print(f"  Best round over 200       : {best_round + 1} (val {best_val:.6f})")
    print(f"  Wall time                 : {elapsed:.0f}s")

    if abs(pct) < 1.0:
        print("\n  -> CONVERGED. Doubling the budget changes validation loss by")
        print(f"     less than 1%. Report in the paper: 'extending FedAvg to 200")
        print(f"     rounds changed mean validation MSE by {abs(pct):.2f}%,")
        print("     confirming convergence at 100 rounds.'")
    else:
        print(f"\n  -> NOT fully converged: {abs(pct):.2f}% further improvement.")
        print("     Report this honestly, or rerun the main experiment at 200")
        print("     rounds (which also doubles the reported communication cost).")
    print("=" * 70)

    out = {
        "seed": SEED, "extended_rounds": EXTENDED_ROUNDS,
        "main_experiment_rounds": cfg.FED_ROUNDS,
        "history": history, "val_at_100": v100, "val_at_200": v200,
        "pct_improvement_100_to_200": pct,
        "best_round": best_round + 1, "best_val": best_val,
    }
    fp = cfg.METRICS_DIR / "convergence_check.json"
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {fp}")


if __name__ == "__main__":
    main()
