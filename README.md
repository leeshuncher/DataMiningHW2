# Money Laundering Is Not an Outlier

Data Mining HW2 (Anomaly Detection). We detect illicit Bitcoin transactions in the
[Elliptic Data Set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) (Weber et al., 2019)
**without using any illicit label for fitting**.

- **Finding:** illicit transactions are *not* outliers. Only 1.7% of them exceed a 3σ rule,
  against 22.3% of licit ones. One-class detectors trained on licit data therefore rank illicit
  transactions *below* normal ones (test ROC-AUC 0.17).
- **Method (Tier 2, LvU):** a positive-unlabeled detector. An XGBoost classifier separates labeled
  licit transactions from unlabeled ones, and its output P(unlabeled | x) is the anomaly score.
  Illicit labels are used only to set the threshold on the validation steps.

Paper: [`paper/main.pdf`](paper/main.pdf). Assignment spec: [`docs/DM-HW2.pdf`](docs/DM-HW2.pdf).

## Results (test steps 35–49, mean ± std over 5 seeds)

| Tier | Method | PR-AUC | ROC-AUC | F1@τ | P@100 |
|---|---|---|---|---|---|
| T1a | Random score | 0.065 ± 0.001 | 0.495 | 0.122 | 0.060 |
| T1b | 3σ rule | 0.053 | 0.326 | 0.010 | 0.040 |
| T2′ | One-class Isolation Forest | 0.038 ± 0.000 | 0.170 | 0.121 | 0.000 |
| **T2** | **LvU classifier (ours, no illicit labels)** | **0.546 ± 0.024** | **0.865** | **0.398** | **0.822** |
| T3a | Random Forest (supervised) | 0.767 ± 0.003 | 0.891 | 0.541 | 1.000 |
| T3b | XGBoost (supervised) | 0.790 ± 0.001 | 0.922 | 0.675 | 1.000 |

Every method collapses after the dark-market shutdown at time step 43. The ablations and the error
analysis are in the paper (§7–§8).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py       # Elliptic CSVs -> data/raw/ (checksum verified)
python scripts/prepare_data.py        # parquet + shared chronological split -> data/processed/
python experiments/eda.py             # §4, training split only
python experiments/run_main.py        # §7.1, 3-tier comparison (~4 min on 4 CPUs)
python experiments/run_ablations.py   # §7.2 (~15 min)
python experiments/analysis.py        # significance, per-step, error analysis, SHAP (~10 min)
python paper/make_tables.py           # LaTeX tables from results/*.json
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The split is chronological by time step and is shared by every method:

| Split | Steps | Role |
|---|---|---|
| train | 1–29 | fitting |
| val | 30–34 | threshold and model selection only |
| test | 35–49 | evaluated once |

Only labeled transactions are evaluated.

## Layout

```
scripts/       download + preprocessing
src/common.py  data loading, methods (all tiers), metrics, threshold rules
experiments/   eda, run_main, run_ablations, analysis
results/       JSON outputs of every experiment (numbers used in the paper)
figures/       figures used in the paper
paper/         ACL-format report (main.tex, refs.bib, generated tables/)
docs/          assignment specification
```

## Data and license

The Elliptic Data Set is released by Elliptic under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). It is not redistributed in
this repository; `scripts/download_data.py` fetches the official Kaggle release (SHA-256 checked)
and falls back to the PyTorch Geometric mirror.
