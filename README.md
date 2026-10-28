# Money Laundering Is Not an Outlier

Data Mining HW2 (Anomaly Detection), Team G13. We detect illicit Bitcoin transactions in the
Elliptic Data Set (Weber et al., 2019) **without using any illicit label for fitting**.
Data category: **Graph**.

- **Finding:** illicit transactions are *not* outliers. Only 1.7% of them exceed a 3σ rule,
  against 22.3% of licit ones. One-class detectors trained on licit data therefore rank illicit
  transactions *below* normal ones (test ROC-AUC 0.17).
- **Method (Tier 2, LvU):** a positive-unlabeled detector. An XGBoost classifier separates labeled
  licit transactions from unlabeled ones, and its output P(unlabeled | x) is the anomaly score.
  Illicit labels are used only to set the threshold on the validation steps.

Report: [`paper/HW2_G13_report.pdf`](paper/HW2_G13_report.pdf). Anonymised draft for the LLM TA
Judge: [`paper/HW2_G13_report_review.pdf`](paper/HW2_G13_report_review.pdf).
Assignment spec: [`docs/DM-HW2.pdf`](docs/DM-HW2.pdf).

## Results (test steps 35–49, mean ± std over 5 seeds)

| Tier | Method | PR-AUC | ROC-AUC | F1@τ | P@100 |
|---|---|---|---|---|---|
| T1a | Random score | 0.065 ± 0.001 | 0.495 | 0.122 | 0.060 |
| T1b | 3σ rule | 0.053 | 0.326 | 0.010 | 0.040 |
| T2′ | One-class Isolation Forest | 0.038 ± 0.000 | 0.170 | 0.121 | 0.000 |
| T2″ | LvU with GraphSAGE message passing | 0.189 ± 0.017 | 0.799 | 0.279 | 0.190 |
| **T2** | **LvU classifier (ours, no illicit labels)** | **0.546 ± 0.024** | **0.865** | **0.398** | **0.822** |
| T3a | Random Forest (supervised) | 0.767 ± 0.003 | 0.891 | 0.541 | 1.000 |
| T3b | XGBoost (supervised) | 0.790 ± 0.001 | 0.922 | 0.675 | 1.000 |
| T3c | GraphSAGE GNN (supervised) | 0.552 ± 0.035 | 0.874 | 0.568 | 0.842 |

- **LvU vs. supervised GraphSAGE:** label-free LvU is statistically indistinguishable from the
  supervised GraphSAGE (p = 0.79).
- **Message passing inside LvU fails:** 59% of the neighbours of illicit transactions are unknown,
  so aggregating them erases the licit/unknown contrast.
- **Dark-market shutdown:** every method collapses after it, at time step 43.

The ablations and the error analysis are in the report (§7–§8).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/download_data.py       # reassemble data/elliptic/ parts -> data/raw/ (SHA-256 checked)
python scripts/prepare_data.py        # parquet + shared chronological split -> data/processed/
python experiments/eda.py             # §4, training split only
python experiments/run_main.py        # §7, 3-tier comparison (~4 min on 4 CPUs)
python experiments/run_gnn.py         # §7, GNNs with PyTorch Geometric (~1.5 h on 4 CPUs; after run_main)
python experiments/run_ablations.py   # §7.1 (~15 min)
python experiments/analysis.py        # significance, per-step, error analysis, SHAP (~10 min)
python paper/make_tables.py           # LaTeX tables from results/*.json
cd paper && for f in HW2_G13_report HW2_G13_report_review; do
  pdflatex $f && bibtex $f && pdflatex $f && pdflatex $f; done
```

`HW2_G13_report_review.tex` builds the anonymised draft from the same source:

- ACL review mode
- Team ID as the only author
- members shown as Member A–H
- no repository link

`scripts/make_submission.sh` packs the E3 upload `HW2_G13.zip` from
`paper/HW2_G13_report.pdf` and `slides/HW2_G13_presentation.pptx` (or `.pdf`).

The split is chronological by time step and is shared by every method:

| Split | Steps | Role |
|---|---|---|
| train | 1–29 | fitting |
| val | 30–34 | threshold and model selection only |
| test | 35–49 | evaluated once |

Only labeled transactions are evaluated.

## Layout

```
data/elliptic/  unmodified Elliptic archive (split into parts), checksums, license note
scripts/        data reassembly/download, preprocessing, submission packing
src/common.py   data loading, tree/rule methods (all tiers), metrics, threshold rules
src/gnn.py      GCN/GraphSAGE detectors (PyTorch Geometric), supervised and LvU roles
experiments/    eda, run_main, run_gnn, run_ablations, analysis
results/        JSON outputs of every experiment (numbers used in the report)
figures/        EDA figure used in the report; per-step F1 and SHAP plots from analysis.py
paper/          ACL-format report (LaTeX source, refs.bib, generated tables/, PDFs)
docs/           assignment specification
```

## Data and license

The Elliptic Data Set is © Elliptic and released under
[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). `data/elliptic/` holds
the official Kaggle archive, unmodified and redistributed for non-commercial coursework; see
[`data/elliptic/README.md`](data/elliptic/README.md). Source:
<https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>.
