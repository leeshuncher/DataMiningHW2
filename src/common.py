"""Shared data loading, methods and metrics for every experiment.

All experiments read the same processed parquet and therefore the same
chronological split (see scripts/prepare_data.py). Evaluation is always on the
labeled rows of the test split; unknown-label transactions are never scored.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEEDS = [0, 1, 2, 3, 4]
LOCAL = [f"local_{i}" for i in range(1, 94)]
AGG = [f"agg_{i}" for i in range(1, 73)]
GRAPH = ["in_degree", "out_degree"]
HOP2 = [f"hop2_{i}" for i in range(1, 73)]  # neighbours' one-hop aggregates, i.e. two hops
FEATURE_SETS = {"LF": LOCAL, "AF": LOCAL + AGG, "AF+deg": LOCAL + AGG + GRAPH, "AF+2hop": LOCAL + AGG + HOP2}

# Review budget used for Precision@k: analysts audit k alerts over the 15-step (~30-week) test period.
REVIEW_BUDGET = 100


# --------------------------------------------------------------------------- data

def load_nodes():
    """Nodes with features, label, split, graph degree and two-hop aggregates."""
    nodes = pd.read_parquet(PROCESSED / "nodes.parquet")
    edges = pd.read_parquet(PROCESSED / "edges.parquet")
    out_deg = edges["src"].value_counts()
    in_deg = edges["dst"].value_counts()
    nodes["out_degree"] = nodes["txId"].map(out_deg).fillna(0).astype("float32")
    nodes["in_degree"] = nodes["txId"].map(in_deg).fillna(0).astype("float32")

    # Two-hop context: mean of the neighbours' one-hop aggregates (undirected; 0 if isolated).
    # Edges never cross time steps, so this uses no information from other periods.
    index = pd.Series(np.arange(len(nodes)), index=nodes["txId"])
    src, dst = index[edges["src"]].to_numpy(), index[edges["dst"]].to_numpy()
    adj = sp.coo_matrix((np.ones(2 * len(src)), (np.r_[src, dst], np.r_[dst, src])),
                        shape=(len(nodes), len(nodes))).tocsr()
    deg = np.asarray(adj.sum(axis=1)).ravel()
    hop2 = sp.diags(1 / np.maximum(deg, 1)) @ adj @ nodes[AGG].to_numpy(dtype="float64")
    nodes = pd.concat([nodes, pd.DataFrame(hop2.astype("float32"), columns=HOP2, index=nodes.index)], axis=1)
    return nodes


def labeled(nodes, split):
    return nodes[(nodes["split"] == split) & (nodes["label"] >= 0)]


def one_class_train(nodes, contamination=0.0, include_unknown=False, seed=0):
    """Training rows for one-class models: licit train rows only by default.

    contamination: fraction of the returned training set made of known illicit
    rows (ablation only). include_unknown: also add unlabeled train rows, i.e.
    the fully unsupervised setting (ablation only).
    """
    train = nodes[nodes["split"] == "train"]
    licit = train[train["label"] == 0]
    parts = [licit]
    if include_unknown:
        parts.append(train[train["label"] == -1])
    if contamination > 0:
        illicit = train[train["label"] == 1]
        n_bad = int(round(contamination * len(licit) / (1 - contamination)))
        n_bad = min(n_bad, len(illicit))
        parts.append(illicit.sample(n=n_bad, random_state=seed))
    return pd.concat(parts)


# --------------------------------------------------------------------------- methods
# Every method exposes fit(X_train, y_train) and score(X) -> higher = more anomalous.

class RandomScore:
    """Tier 1a: uniform random score (performance floor; PR-AUC ~ prevalence)."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y=None):
        return self

    def score(self, X):
        return self.rng.random(len(X))


class MaxZScore:
    """Tier 1b: 3-sigma rule. Score = max_j |z_j| w.r.t. licit training statistics."""

    def fit(self, X, y=None):
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0) + 1e-9
        return self

    def score(self, X):
        return np.abs((X - self.mu) / self.sd).max(axis=1)


class IForest:
    """Tier 2: Isolation Forest fit on licit-only rows; s(x) = -score_samples(x)."""

    def __init__(self, seed=0, n_estimators=200, max_samples=256, max_features=1.0):
        self.model = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples, max_features=max_features,
            random_state=seed, n_jobs=-1)

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def score(self, X):
        return -self.model.score_samples(X)


class LicitVsUnlabeled:
    """Tier 2 (ours): positive-unlabeled detector.

    A classifier learns to tell labeled licit rows (class 0) from unlabeled rows
    (class 1); s(x) = P(unlabeled | x). Illicit labels are never used for fitting:
    the score is high where unlabeled mass is dense but licit mass is not, which
    is where the illicit clusters live (Elkan & Noto, 2008).
    """

    def __init__(self, seed=0, n_estimators=300, max_depth=6):
        self.model = XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="logloss")

    def fit(self, X_licit, X_unlabeled):
        X = np.vstack([X_licit, X_unlabeled])
        t = np.r_[np.zeros(len(X_licit)), np.ones(len(X_unlabeled))]
        self.model.fit(X, t)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


def pu_train(nodes, contamination=0.0, unlabeled_frac=1.0, seed=0):
    """(licit rows, unlabeled rows) from the train split for LicitVsUnlabeled.

    contamination: fraction of the licit side replaced by known illicit rows,
    i.e. label noise in the normal class (ablation only).
    unlabeled_frac: subsample of the unlabeled pool (ablation only).
    """
    train = nodes[nodes["split"] == "train"]
    licit = train[train["label"] == 0]
    unlabeled = train[train["label"] == -1]
    if unlabeled_frac < 1.0:
        unlabeled = unlabeled.sample(frac=unlabeled_frac, random_state=seed)
    if contamination > 0:
        n_bad = int(round(contamination * len(licit)))
        bad = train[train["label"] == 1].sample(n=n_bad, random_state=seed)
        licit = pd.concat([licit.sample(n=len(licit) - n_bad, random_state=seed), bad])
    return licit, unlabeled


class Mahalanobis:
    """Scoring-function ablation: Mahalanobis distance to the licit training mean."""

    def fit(self, X, y=None):
        cov = LedoitWolf().fit(X)
        self.mu = cov.location_
        self.prec = cov.precision_
        return self

    def score(self, X):
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.prec, d)


class RandomForest:
    """Tier 3a: supervised Random Forest, configured as in Weber et al. (2019)."""

    def __init__(self, seed=0):
        self.model = RandomForestClassifier(
            n_estimators=50, max_features=50, random_state=seed, n_jobs=-1)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBoost:
    """Tier 3b: supervised gradient-boosted trees."""

    def __init__(self, seed=0):
        self.model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, eval_metric="aucpr")

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X):
        return self.model.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- thresholds & metrics

def best_f1_threshold(y, s):
    """tau maximizing F1 on a labeled validation set (alert if s >= tau)."""
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def min_cost_threshold(y, s, cost_ratio):
    """tau minimizing cost_ratio * FN + FP on a labeled validation set (alert if s >= tau).

    cost_ratio = c_FN / c_FP, the cost of a missed illicit transaction in units of one
    false alert. Returns inf (no alerts) when alerting on nothing is cheapest.
    """
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    cost = cost_ratio * (ys.sum() - tp) + fp  # cost of alerting on the top-k, k = 1..n
    boundary = np.r_[ss[1:] != ss[:-1], True]  # ties share one threshold
    k = int(np.argmin(np.where(boundary, cost, np.inf)))
    return float(ss[k]) if cost[k] < cost_ratio * ys.sum() else float("inf")


def precision_at_k(y, s, k=REVIEW_BUDGET):
    top = np.argsort(-s, kind="stable")[:k]
    return float(np.mean(y[top]))


def evaluate(y, s, tau):
    pred = (s >= tau).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)) if np.ptp(s) > 0 else 0.5,
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(pred[y == 1].sum() / max(pred.sum(), 1)),
        "recall": float(pred[y == 1].mean()),
        "p_at_k": precision_at_k(y, s),
        "alerts": int(pred.sum()),
    }


def timed(fn, *args):
    start = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - start


def save_json(obj, name):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=2, default=float) + "\n")
