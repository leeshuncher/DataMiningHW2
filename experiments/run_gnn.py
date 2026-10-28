"""Graph neural networks on the shared split (§7): supervised GNN (Tier 3c) and
the graph version of our detector (LvU-GNN, licit vs unknown targets).

1. Model selection on the validation split (seed 0): convolution, width, features.
2. Every role x 5 seeds: tau = best F1 on val, evaluated once on test.
3. Significance of the PR-AUC gap to LvU (paired t-test over seeds, bootstrap CI).

Outputs: results/gnn.json
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score

from common import FEATURE_SETS, RESULTS, SEEDS, best_f1_threshold, evaluate, load_nodes, save_json
from gnn import GNNDetector, edge_index

OURS = "T2 Licit-vs-Unlabeled (ours)"
ROLES = {"sup": "T3c GNN (sup.)", "pu": "T2'' LvU-GNN"}
SHUTDOWN_STEP = 43
N_BOOT = 1000


def select(nodes, ei):
    grid = {
        "sup": [{"conv": c, "hidden": h, "features": "AF"} for c, h in itertools.product(["gcn", "sage"], [64, 128])],
        "pu": [{"conv": c, "hidden": 64, "features": f} for c, f in itertools.product(["gcn", "sage"], ["LF", "AF"])],
    }
    chosen, log = {}, {}
    for role, cfgs in grid.items():
        log[role] = []
        for cfg in cfgs:
            det = GNNDetector(role, cfg["conv"], cfg["hidden"], seed=0)
            det.fit_score(nodes, ei, FEATURE_SETS[cfg["features"]])
            log[role].append({**cfg, "val_pr_auc": round(det.val_pr_auc, 4), "best_epoch": det.best_epoch})
            print(role, log[role][-1], flush=True)
        chosen[role] = max(log[role], key=lambda r: r["val_pr_auc"])
    return chosen, log


def significance(y, ours_runs, ours_S, runs, S):
    shared = sorted(set(ours_runs) & set(runs))
    t = ttest_rel([ours_runs[s] for s in shared], [runs[s] for s in shared])
    rng = np.random.default_rng(0)
    gaps = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        a = np.mean([average_precision_score(y[i], ours_S[i, j]) for j in range(ours_S.shape[1])])
        b = np.mean([average_precision_score(y[i], S[i, j]) for j in range(S.shape[1])])
        gaps.append(a - b)
    return {"gap_mean": float(np.mean(list(ours_runs.values())) - np.mean(list(runs.values()))),
            "paired_t_p": float(t.pvalue),
            "boot_ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))]}


def main():
    nodes = load_nodes()
    ei = edge_index(nodes)
    split, y_all, steps_all = nodes["split"].to_numpy(), nodes["label"].to_numpy(), nodes["time_step"].to_numpy()
    val, test = (split == "val") & (y_all >= 0), (split == "test") & (y_all >= 0)
    y, steps = y_all[test], steps_all[test]

    chosen, log = select(nodes, ei)

    main_res = json.loads((RESULTS / "main.json").read_text())
    scores = pd.read_parquet(RESULTS / "scores_test.parquet")
    assert (scores["txId"].to_numpy() == nodes.loc[test, "txId"].to_numpy()).all()
    ours_runs = {r["seed"]: r["pr_auc"] for r in main_res[OURS]["runs"]}
    ours_S = scores[[c for c in scores.columns if c.startswith(OURS + "|")]].to_numpy()

    out = {"chosen": chosen, "grid": log}
    for role, name in ROLES.items():
        cfg = chosen[role]
        runs, S = [], []
        for seed in SEEDS:
            det = GNNDetector(role, cfg["conv"], cfg["hidden"], seed=seed)
            s = det.fit_score(nodes, ei, FEATURE_SETS[cfg["features"]])
            tau = best_f1_threshold(y_all[val], s[val])
            m = evaluate(y, s[test], tau)
            per_period = {p: float(average_precision_score(y[mask], s[test][mask]))
                          for p, mask in [("before", steps < SHUTDOWN_STEP), ("from", steps >= SHUTDOWN_STEP)]}
            per_step = {int(t): evaluate(y[steps == t], s[test][steps == t], tau) for t in np.unique(steps)}
            runs.append({"seed": seed, "tau": tau, "best_epoch": det.best_epoch, "fit_s": det.fit_s,
                         "ms_per_sample": det.ms_per_sample, **m, "pr_auc_period": per_period, "per_step": per_step})
            S.append(s[test])
            print(name, seed, {k: round(v, 3) for k, v in m.items()}, flush=True)
        S = np.array(S).T
        keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "p_at_k", "alerts", "fit_s", "ms_per_sample", "tau"]
        summary = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                   for k in keys}
        summary["pr_auc_before"] = float(np.mean([r["pr_auc_period"]["before"] for r in runs]))
        summary["pr_auc_from"] = float(np.mean([r["pr_auc_period"]["from"] for r in runs]))
        sig = significance(y, ours_runs, ours_S, {r["seed"]: r["pr_auc"] for r in runs}, S)
        out[name] = {"config": cfg, "summary": summary, "significance_vs_ours": sig, "runs": runs}
        print(name, "vs ours", sig, flush=True)

    save_json(out, "gnn.json")


if __name__ == "__main__":
    main()
