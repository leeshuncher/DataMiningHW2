"""Main 3-tier comparison (§7.1) on the shared chronological split.

1. Model selection for Tier 2 on the validation split (seed 0).
2. Every method x 5 seeds: fit on train, pick tau = best F1 on val, evaluate once on test.

Outputs: results/hparams.json, results/main.json, results/scores_test.parquet
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from common import (FEATURE_SETS, RESULTS, SEEDS, IForest, LicitVsUnlabeled, MaxZScore,
                    RandomForest, RandomScore, XGBoost, best_f1_threshold, evaluate, labeled, load_nodes,
                    one_class_train, pu_train, save_json, timed)


def fit_method(name, nodes, cfg, seed):
    """Fit one method on the train split; returns the fitted model."""
    cols = FEATURE_SETS[cfg["features"]]
    if name in ("T1a random", "T1b 3-sigma", "T2' IForest (one-class)"):
        X = one_class_train(nodes)[cols].to_numpy()
        model = {"T1a random": RandomScore(seed), "T1b 3-sigma": MaxZScore(),
                 "T2' IForest (one-class)": IForest(seed=seed, **cfg.get("params", {}))}[name]
        return model.fit(X)
    if name == "T2 Licit-vs-Unlabeled (ours)":
        licit, unlabeled = pu_train(nodes, seed=seed)
        return LicitVsUnlabeled(seed=seed, **cfg.get("params", {})).fit(
            licit[cols].to_numpy(), unlabeled[cols].to_numpy())
    train = labeled(nodes, "train")
    model = {"T3a RandomForest (sup.)": RandomForest, "T3b XGBoost (sup.)": XGBoost}[name](seed=seed)
    return model.fit(train[cols].to_numpy(), train["label"].to_numpy())


def select_tier2(nodes):
    """Pick feature set and hyperparameters of the Tier 2 models by validation PR-AUC."""
    val = labeled(nodes, "val")
    grid = {
        "T2 Licit-vs-Unlabeled (ours)": [
            {"features": f, "params": {"max_depth": d, "n_estimators": n}}
            for f, d, n in itertools.product(["LF", "AF", "AF+deg"], [4, 6, 8], [100, 300])],
        "T2' IForest (one-class)": [
            {"features": f, "params": {"max_samples": m}}
            for f, m in itertools.product(["LF", "AF", "AF+deg"], [256, 2048])],
    }
    chosen, log = {}, {}
    for name, cfgs in grid.items():
        log[name] = []
        for cfg in cfgs:
            model = fit_method(name, nodes, cfg, seed=0)
            s = model.score(val[FEATURE_SETS[cfg["features"]]].to_numpy())
            ap = average_precision_score(val["label"], s)
            log[name].append({**cfg, "val_pr_auc": round(float(ap), 4)})
            print(name, cfg, round(ap, 4))
        chosen[name] = max(log[name], key=lambda r: r["val_pr_auc"])
    return chosen, log


def main():
    nodes = load_nodes()
    val, test = labeled(nodes, "val"), labeled(nodes, "test")

    chosen, log = select_tier2(nodes)
    save_json({"chosen": chosen, "grid": log}, "hparams.json")

    configs = {
        "T1a random": {"features": "LF"},
        "T1b 3-sigma": {"features": "LF"},
        "T2' IForest (one-class)": chosen["T2' IForest (one-class)"],
        "T2 Licit-vs-Unlabeled (ours)": chosen["T2 Licit-vs-Unlabeled (ours)"],
        "T3a RandomForest (sup.)": {"features": "AF"},  # Weber et al. (2019): RF on all features
        "T3b XGBoost (sup.)": {"features": "AF"},
    }
    deterministic = {"T1b 3-sigma"}

    results, scores = {}, {"txId": test["txId"].to_numpy(), "time_step": test["time_step"].to_numpy(),
                           "label": test["label"].to_numpy()}
    for name, cfg in configs.items():
        cols = FEATURE_SETS[cfg["features"]]
        runs = []
        for seed in ([0] if name in deterministic else SEEDS):
            model, fit_s = timed(fit_method, name, nodes, cfg, seed)
            tau = best_f1_threshold(val["label"].to_numpy(), model.score(val[cols].to_numpy()))
            if name == "T1b 3-sigma":
                tau = 3.0  # fixed rule, no tuning
            X_test = test[cols].to_numpy()
            s, score_s = timed(model.score, X_test)
            y = test["label"].to_numpy()
            m = evaluate(y, s, tau)
            per_step = {}
            for step in sorted(test["time_step"].unique()):
                mask = test["time_step"].to_numpy() == step
                per_step[int(step)] = evaluate(y[mask], s[mask], tau) | {"n_illicit": int(y[mask].sum())}
            runs.append({"seed": seed, "tau": tau, "fit_s": fit_s,
                         "ms_per_sample": 1000 * score_s / len(X_test), **m, "per_step": per_step})
            scores[f"{name}|{seed}"] = s
            print(name, seed, {k: round(v, 3) for k, v in m.items()})
        summary = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))}
                   for k in ["pr_auc", "roc_auc", "f1", "precision", "recall", "p_at_k", "alerts",
                             "fit_s", "ms_per_sample", "tau"]}
        results[name] = {"config": cfg, "summary": summary, "runs": runs}

    save_json(results, "main.json")
    pd.DataFrame(scores).to_parquet(RESULTS / "scores_test.parquet", index=False)


if __name__ == "__main__":
    main()
