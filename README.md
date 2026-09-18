# Who Benefits from Federation?

Code and results for **"Who Benefits from Federation? Heterogeneous Gains in
Cross-Operator Network Traffic Representation Learning"**, 13th International
Workshop on Innovating the Network for Data-Intensive Science (INDIS 2026),
held in conjunction with SC26.

Mohammad Arafath Uddin Shariff, Byrav Ramamurthy
School of Computing, University of Nebraska-Lincoln

## Overview

We evaluate federated representation learning across 20 heterogeneous network
clients drawn from a national research backbone (Internet2, 10 routers) and a
European NREN (CESNET3, 10 academic institutions), with a commercial regional
ISP held out as an unseen-network probe. Each client trains an autoencoder
over 24-hour traffic windows, and we compare centralized, federated (FedAvg),
and local-only training at equal optimization budget across five seeds.

Two findings:

1. Federated training is **statistically equivalent** to centralized pooling
   (two one-sided tests, +/-5% bound, p = 0.020). The privacy constraint costs
   nothing in representation quality.
2. The benefit of federation is **not uniform**. Backbone clients gain a median
   9.2% reduction in reconstruction error; campus clients gain 1.9%
   (Mann-Whitney p = 0.0003). The effect survives outlier removal and is not
   explained by training-set size. Two independent manipulations, proximal
   regularization strength and local-epoch budget, each selectively suppress
   the benefit for irregular-traffic clients, identifying unconstrained local
   adaptation as the mechanism.

## What can and cannot be reproduced

| Source | Status |
|---|---|
| Internet2 | **Restricted.** NDA-governed NetFlow. Hourly series cannot be redistributed. |
| CESNET3 | **Public.** [CESNET-TimeSeries24](https://doi.org/10.5281/zenodo.13382427) |
| Regional ISP | **Restricted.** 48-hour snapshot, not redistributable. |

Because two of three sources are access-restricted, this repository
deliberately excludes:

- **Client tensors** (`*.npz`) — these store per-client scaler parameters and
  are therefore invertible back to raw packet counts.
- **Trained checkpoints** (`*.pt`) — trained on restricted data.
- **Cached embeddings** — derived from restricted data.

What **is** included: the complete pipeline, and every metric table underlying
every figure and table in the paper, under `results/metrics/`. Anyone can
verify the reported numbers without access to the raw data. The pipeline
accepts any equivalently formatted hourly series, so alternative sources can
be substituted for the restricted clients.

Full reproduction from raw data requires the public CESNET corpus plus
equivalently formatted hourly series for the remaining clients.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Edit the three source paths at the top of `fed_config.py` before running.

## Pipeline

Scripts use `# %%` cell markers: open directly as notebooks in VSCode or
JupyterLab (via jupytext), or run as plain scripts. Run in order.

| Script | Purpose | Runtime |
|---|---|---|
| `01_build_clients.py` | Partition, normalize, window; Table I, Fig. 2 | seconds |
| `02_train_regimes.py` | Train three regimes, 5 seeds | ~15 min |
| `03_analyze_figures.py` | Per-client metrics, embeddings, Table III | ~2 min |
| `03b_corrections.py` | Corrected convergence/per-client figures and tables | seconds |
| `04_stratified_benefit.py` | The central analysis; Fig. 4 | seconds |
| `05_ablations.py` | FedProx sweep, window and local-epoch ablations | ~2.3 h |
| `06_final_statistics.py` | TOST equivalence; Fig. 5 | seconds |
| `07_convergence_check.py` | 200-round verification | ~4 min |

`03b_corrections.py` must run **after** `03_analyze_figures.py`.

Total end-to-end: roughly 3 hours on a single NVIDIA H200.

## Design notes

**Derived configuration.** Quantities computable from others are computed,
never duplicated. Client eligibility follows
`T_min = ceil((N_min + L - 1) / min(rho_val, rho_test))`, so the admission
threshold and window configuration cannot silently disagree.

**Cache validation.** Each stage writes a configuration signature alongside its
artifacts and verifies it on load, so results produced under different settings
are detected rather than reused.

**Split before window.** Windows are extracted independently within each
chronological partition. Windowing first would leak 23 of 24 hours across the
train/test boundary at unit stride.

**Per-client normalization.** Scaling statistics are fitted per client on the
training partition only. A federation-wide scaler would require exchanging
distributional statistics, the exact disclosure the protocol avoids.

## Reproducibility notes

Seeds {42..46} govern initialization, batch ordering, and per-client model
initialization. Training is seeded but **not bit-reproducible across differing
GPU hardware**, since kernel selection and reduction order vary by device. The
12-hour window configuration is the most sensitive in this respect; the paper
reports it as not statistically significant (p = 0.154).

## Citation

```bibtex
@inproceedings{shariff2026whobenefits,
  title     = {Who Benefits from Federation? Heterogeneous Gains in
               Cross-Operator Network Traffic Representation Learning},
  author    = {Shariff, Mohammad Arafath Uddin and Ramamurthy, Byrav},
  booktitle = {Proc. IEEE/ACM Innovating the Network for Data-Intensive
               Science (INDIS)},
  year      = {2026}
}
```

## Acknowledgments

We thank the Internet2 consortium and the CESNET association for providing the
network telemetry underlying this work, and the regional Internet service
provider whose traffic snapshot served as our held-out evaluation network.
Supported in part by NSF grants OAC-2322369 and CNS-2413851, and DOE grant
DE-SC0024648.
