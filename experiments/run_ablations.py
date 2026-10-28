"""Controlled ablations of the Tier 2 detector (§7.1). One factor varies at a time;
everything else stays at the configuration selected in run_main.py.

A1 contamination   share of illicit rows hidden in the licit side: 0/1/5/10 %
A2 unlabeled pool  fraction of the unlabeled train pool used: 5/25/50/100 %
A3 scoring family  label-free detectors on the same features
A4 feature set     LF / AF / AF+deg / AF+2hop
A5 threshold rule  best-F1 on val / 95th pct of licit-train scores / weekly review budget

Output: results/ablations.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

from common import (FEATURE_SETS, RESULTS, SEEDS, IForest, LicitVsUnlabeled, Mahalanobis,
                    best_f1_threshold, evaluate, labeled, load_nodes, one_class_train, pu_train,
                    save_json)

WEEKLY_BUDGET = 30  # alerts analysts can review per time step (~2 weeks of data)


class LOFNovelty:
    def __init__(self, seed=0):
        self.scaler = StandardScaler()
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)

    def fit(self, X):
        self.lof.fit(self.scaler.fit_transform(X))
        return self

    def score(self, X):
        return -self.lof.score_samples(self.scaler.transform(X))


class KNNDensityRatio:
    """PU baseline without a classifier: log d_k(x, licit) - log d_k(x, unlabeled)."""

    def __init__(self, k=10):
        self.k = k

    def fit(self, X_licit, X_unlabeled):
        self.scaler = StandardScaler().fit(X_licit)
        self.nn_l = NearestNeighbors(n_neighbors=self.k).fit(self.scaler.transform(X_licit))
        self.nn_u = NearestNeighbors(n_neighbors=self.k).fit(self.scaler.transform(X_unlabeled))
        return self

    def score(self, X):
        Z = self.scaler.transform(X)
        dl = self.nn_l.kneighbors(Z)[0].mean(axis=1)
        du = self.nn_u.kneighbors(Z)[0].mean(axis=1)
        return np.log(dl + 1e-6) - np.log(du + 1e-6)


def run(nodes, cols, build, seeds, tau_rule="best_f1"):
    """build(seed) -> fitted model. Returns mean/std of test metrics over seeds."""
    val, test = labeled(nodes, "val"), labeled(nodes, "test")
    y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
    runs = []
    for seed in seeds:
        model = build(seed)
        s_val, s_test = model.score(val[cols].to_numpy()), model.score(test[cols].to_numpy())
        if tau_rule == "best_f1":
            tau = best_f1_threshold(y_val, s_val)
            m = evaluate(y_test, s_test, tau)
        elif tau_rule == "licit_p95":
            licit_train = one_class_train(nodes)[cols].to_numpy()
            tau = float(np.percentile(model.score(licit_train), 95))
            m = evaluate(y_test, s_test, tau)
        elif tau_rule == "weekly_budget":
            # Top-B scores of every time step become alerts (scored over all test rows of that step).
            pred = np.zeros_like(y_test)
            steps = test["time_step"].to_numpy()
            for step in np.unique(steps):
                idx = np.where(steps == step)[0]
                pred[idx[np.argsort(-s_test[idx])[:WEEKLY_BUDGET]]] = 1
            tau = None
            m = evaluate(y_test, s_test + 1e9 * pred, 1e9)  # alerts = budgeted rows
        runs.append(m | {"tau": tau})
    keys = ["pr_auc", "roc_auc", "f1", "precision", "recall", "p_at_k", "alerts"]
    return {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
            for k in keys}


def main():
    nodes = load_nodes()
    chosen = json.loads((RESULTS / "hparams.json").read_text())["chosen"]
    cfg = chosen["T2 Licit-vs-Unlabeled (ours)"]
    feat, params = cfg["features"], cfg["params"]
    cols = FEATURE_SETS[feat]

    def pu(seed, contamination=0.0, unlabeled_frac=1.0, cols=cols):
        licit, unlabeled = pu_train(nodes, contamination, unlabeled_frac, seed)
        return LicitVsUnlabeled(seed=seed, **params).fit(licit[cols].to_numpy(), unlabeled[cols].to_numpy())

    out = {"config": cfg}

    out["A1_contamination"] = {}
    for c in [0.0, 0.01, 0.05, 0.10]:
        out["A1_contamination"][f"{c:.0%}"] = run(nodes, cols, lambda s, c=c: pu(s, contamination=c), SEEDS)
        print("A1", c, out["A1_contamination"][f"{c:.0%}"]["pr_auc"])

    out["A2_unlabeled_pool"] = {}
    for frac in [0.05, 0.25, 0.5, 1.0]:
        out["A2_unlabeled_pool"][f"{frac:.0%}"] = run(nodes, cols, lambda s, f=frac: pu(s, unlabeled_frac=f), SEEDS)
        print("A2", frac, out["A2_unlabeled_pool"][f"{frac:.0%}"]["pr_auc"])

    licit_only = one_class_train(nodes)[cols].to_numpy()
    lic, unl = pu_train(nodes)
    out["A3_scoring"] = {
        "One-class IForest": run(nodes, cols, lambda s: IForest(seed=s).fit(licit_only), SEEDS),
        "One-class Mahalanobis": run(nodes, cols, lambda s: Mahalanobis().fit(licit_only), [0]),
        "One-class LOF": run(nodes, cols, lambda s: LOFNovelty().fit(licit_only), [0]),
        "PU kNN density ratio": run(nodes, cols, lambda s: KNNDensityRatio().fit(
            lic[cols].to_numpy(), unl[cols].to_numpy()), [0]),
        "PU classifier (ours)": run(nodes, cols, pu, SEEDS),
    }
    print("A3", {k: v["pr_auc"]["mean"] for k, v in out["A3_scoring"].items()})

    out["A4_features"] = {}
    for fs in ["LF", "AF", "AF+deg", "AF+2hop"]:
        out["A4_features"][fs] = run(nodes, FEATURE_SETS[fs],
                                     lambda s, c=FEATURE_SETS[fs]: pu(s, cols=c), SEEDS)
        print("A4", fs, out["A4_features"][fs]["pr_auc"])

    out["A5_threshold"] = {rule: run(nodes, cols, pu, SEEDS, tau_rule=rule)
                           for rule in ["best_f1", "licit_p95", "weekly_budget"]}
    print("A5", {k: v["f1"]["mean"] for k, v in out["A5_threshold"].items()})

    save_json(out, "ablations.json")


if __name__ == "__main__":
    main()
