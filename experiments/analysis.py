"""Significance tests, temporal breakdown, error analysis and interpretability (§7-§8).

Reads results/main.json + results/scores_test.parquet written by run_main.py.
Outputs: results/analysis.json, figures/per_step.pdf, figures/shap.pdf
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ttest_rel
from sklearn.metrics import average_precision_score

from common import (FEATURE_SETS, FIGURES, RESULTS, SEEDS, LicitVsUnlabeled, labeled, load_nodes,
                    one_class_train, pu_train, save_json)

plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
OURS = "T2 Licit-vs-Unlabeled (ours)"
SHUTDOWN_STEP = 43  # dark-market shutdown reported by Weber et al. (2019)
N_BOOT = 1000


def seed_mean_scores(scores, name):
    cols = [c for c in scores.columns if c.startswith(name + "|")]
    return scores[cols].to_numpy(), cols


def significance(main, scores):
    """Paired t-test over seeds and bootstrap 95% CI of the test PR-AUC gap vs ours."""
    y = scores["label"].to_numpy()
    ours_runs = {r["seed"]: r["pr_auc"] for r in main[OURS]["runs"]}
    S_ours, _ = seed_mean_scores(scores, OURS)
    rng = np.random.default_rng(0)
    boot_idx = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]
    out = {}
    for name, res in main.items():
        if name == OURS:
            continue
        other = {r["seed"]: r["pr_auc"] for r in res["runs"]}
        shared = sorted(set(ours_runs) & set(other))
        t = ttest_rel([ours_runs[s] for s in shared], [other[s] for s in shared]) if len(shared) > 1 else None
        S_other, _ = seed_mean_scores(scores, name)
        gaps = []
        for idx in boot_idx:
            yb = y[idx]
            a = np.mean([average_precision_score(yb, S_ours[idx, j]) for j in range(S_ours.shape[1])])
            b = np.mean([average_precision_score(yb, S_other[idx, j]) for j in range(S_other.shape[1])])
            gaps.append(a - b)
        out[name] = {
            "gap_mean": float(np.mean([ours_runs[s] for s in ours_runs]) - res["summary"]["pr_auc"]["mean"]),
            "paired_t_p": float(t.pvalue) if t is not None else None,
            "boot_ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
        }
        print("sig", name, out[name])
    return out


def pre_post_shutdown(scores):
    """Test PR-AUC (seed mean) on steps before vs from the dark-market shutdown."""
    y, steps = scores["label"].to_numpy(), scores["time_step"].to_numpy()
    out = {}
    for name in sorted({c.split("|")[0] for c in scores.columns if "|" in c}):
        S, _ = seed_mean_scores(scores, name)
        out[name] = {
            period: float(np.mean([average_precision_score(y[mask], S[mask, j]) for j in range(S.shape[1])]))
            for period, mask in [("before", steps < SHUTDOWN_STEP), ("from", steps >= SHUTDOWN_STEP)]}
        out[name]["prevalence_before"] = float(y[steps < SHUTDOWN_STEP].mean())
        out[name]["prevalence_from"] = float(y[steps >= SHUTDOWN_STEP].mean())
    return out


def per_step_figure(main):
    names = {OURS: ("T2 LvU (ours)", "#E45756", "-"),
             "T3a RandomForest (sup.)": ("T3a RF (sup.)", "#4C78A8", "--"),
             "T3b XGBoost (sup.)": ("T3b XGB (sup.)", "#72B7B2", "--"),
             "T2' IForest (one-class)": ("One-class IF", "#B279A2", ":")}
    fig, ax = plt.subplots(figsize=(3.1, 1.85))
    steps = sorted(int(s) for s in main[OURS]["runs"][0]["per_step"])
    counts = [main[OURS]["runs"][0]["per_step"][str(s)]["n_illicit"] for s in steps]
    ax2 = ax.twinx()
    ax2.bar(steps, counts, color="lightgray", width=0.7)
    ax2.set_ylabel("# illicit tx in step", color="gray")
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    out = {}
    for name, (label, color, ls) in names.items():
        f1 = np.array([[r["per_step"][str(s)]["f1"] for s in steps] for r in main[name]["runs"]])
        ax.plot(steps, f1.mean(axis=0), ls, color=color, label=label, marker="o", ms=2)
        out[name] = {"f1_before_shutdown": float(f1[:, [s < SHUTDOWN_STEP for s in steps]].mean()),
                     "f1_from_shutdown": float(f1[:, [s >= SHUTDOWN_STEP for s in steps]].mean())}
    ax.axvline(SHUTDOWN_STEP - 0.5, color="black", lw=0.8)
    ax.text(SHUTDOWN_STEP - 0.3, 0.95, "dark-market\nshutdown", fontsize=6, va="top")
    ax.set_xlabel("test time step (1 step ≈ 2 weeks)")
    ax.set_ylabel("illicit F1 (mean of seeds)")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6, loc="center right", bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout()
    fig.savefig(FIGURES / "per_step.pdf")
    return out


def shap_values(model, X):
    booster = model.model.get_booster()
    contrib = booster.predict(xgb.DMatrix(X), pred_contribs=True)
    return contrib[:, :-1], contrib[:, -1]  # per-feature log-odds contributions, bias


def error_analysis(nodes, main, scores):
    cfg = main[OURS]["config"]
    cols = FEATURE_SETS[cfg["features"]]
    licit, unlabeled = pu_train(nodes, seed=0)
    model = LicitVsUnlabeled(seed=0, **cfg["params"]).fit(licit[cols].to_numpy(), unlabeled[cols].to_numpy())
    run0 = next(r for r in main[OURS]["runs"] if r["seed"] == 0)
    tau = run0["tau"]

    test = labeled(nodes, "test").reset_index(drop=True)
    s = model.score(test[cols].to_numpy())
    assert np.allclose(s, scores[f"{OURS}|0"].to_numpy(), atol=1e-5)
    y = test["label"].to_numpy()
    sup, _ = seed_mean_scores(scores, "T3a RandomForest (sup.)")
    sup_tau = np.mean([r["tau"] for r in main["T3a RandomForest (sup.)"]["runs"]])
    phi, bias = shap_values(model, test[cols].to_numpy())
    licit_median = one_class_train(nodes)[cols].median()

    def describe(i, kind):
        top = np.argsort(-np.abs(phi[i]))[:3]
        return {"type": kind, "txId": int(test.loc[i, "txId"]), "time_step": int(test.loc[i, "time_step"]),
                "score": float(s[i]), "tau": tau, "margin": float(s[i] - tau),
                "rf_score": float(sup[i].mean()), "rf_flags": bool(sup[i].mean() >= sup_tau),
                "in_degree": float(test.loc[i, "in_degree"]), "out_degree": float(test.loc[i, "out_degree"]),
                "top_features": [{"feature": cols[j], "value": float(test.loc[i, cols[j]]),
                                  "licit_median": float(licit_median[cols[j]]), "shap": float(phi[i, j])}
                                 for j in top]}

    fn = np.where((y == 1) & (s < tau))[0]
    fp = np.where((y == 0) & (s >= tau))[0]
    tp = np.where((y == 1) & (s >= tau))[0]
    cases = [
        describe(fn[np.argmin(s[fn])], "FN (inversion)"),
        describe(fn[np.argmax(s[fn])], "FN (near miss)"),
        describe(fp[np.argmax(s[fp])], "FP (confident)"),
        describe(fp[np.argmin(s[fp])], "FP (near miss)"),
        describe(tp[np.argmax(s[tp])], "TP (explained)"),
    ]
    steps = test["time_step"].to_numpy()
    fn_profile = {
        "n_fn": int(len(fn)), "n_fp": int(len(fp)), "n_tp": int(len(tp)),
        "fn_share_from_shutdown": float(np.mean(steps[fn] >= SHUTDOWN_STEP)),
        "illicit_share_from_shutdown": float(np.mean(steps[y == 1] >= SHUTDOWN_STEP)),
        "fn_also_missed_by_rf": float(np.mean(sup[fn].mean(axis=1) < sup_tau)),
        "fn_score_quartiles": np.percentile(s[fn], [25, 50, 75]).tolist(),
        "fn_far_below_tau_share": float(np.mean(s[fn] < tau / 2)),
        "fp_zero_in_degree_share": float(np.mean(test.loc[fp, "in_degree"] == 0)),
        "licit_zero_in_degree_share": float(np.mean(test.loc[y == 0, "in_degree"] == 0)),
    }

    # Global importance on test illicit rows (reported in the text) + one local explanation (figure).
    mean_abs = np.abs(phi[y == 1]).mean(axis=0)
    top = np.argsort(-mean_abs)[:8]
    fig, ax = plt.subplots(figsize=(3.1, 1.6))
    i = tp[np.argmax(s[tp])]
    loc = np.argsort(-np.abs(phi[i]))[:6]
    ax.barh([f"{cols[j]}={test.loc[i, cols[j]]:.2f}" for j in loc][::-1], phi[i, loc][::-1],
            color=["#E45756" if v > 0 else "#4C78A8" for v in phi[i, loc][::-1]])
    ax.set_xlabel("SHAP contribution (log-odds)")
    ax.set_title(f"flagged tx {int(test.loc[i, 'txId'])}, s={s[i]:.3f}")
    fig.tight_layout()
    fig.savefig(FIGURES / "shap.pdf")

    global_imp = [{"feature": cols[j], "mean_abs_shap": float(mean_abs[j])} for j in top]
    return {"tau": tau, "cases": cases, "profile": fn_profile, "global_importance": global_imp}


def practical_impact(main, scores):
    """Translate scores into analyst workload at the val-tuned threshold and a fixed budget."""
    y = scores["label"].to_numpy()
    out = {}
    for name in [OURS, "T3a RandomForest (sup.)", "T2' IForest (one-class)"]:
        r = main[name]["summary"]
        out[name] = {"alerts": r["alerts"]["mean"], "precision": r["precision"]["mean"],
                     "recall": r["recall"]["mean"], "caught": r["recall"]["mean"] * y.sum(),
                     "p_at_100": r["p_at_k"]["mean"]}
    out["test_illicit"] = int(y.sum())
    out["test_prevalence"] = float(y.mean())
    return out


def main():
    FIGURES.mkdir(exist_ok=True)
    main_res = json.loads((RESULTS / "main.json").read_text())
    scores = pd.read_parquet(RESULTS / "scores_test.parquet")
    nodes = load_nodes()
    out = {
        "significance": significance(main_res, scores),
        "per_step": per_step_figure(main_res),
        "pr_auc_shutdown": pre_post_shutdown(scores),
        "error_analysis": error_analysis(nodes, main_res, scores),
        "impact": practical_impact(main_res, scores),
    }
    save_json(out, "analysis.json")
    print(json.dumps(out["error_analysis"]["profile"], indent=1))


if __name__ == "__main__":
    main()
