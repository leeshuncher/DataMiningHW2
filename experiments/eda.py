"""Exploratory feature analysis (§4). Uses the TRAINING split only.

Outputs: results/eda.json, figures/eda_*.pdf
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif

from common import AGG, FIGURES, GRAPH, HOP2, LOCAL, load_nodes, save_json

plt.rcParams.update({"font.size": 8, "figure.dpi": 150})
LICIT_C, ILLICIT_C = "#4C78A8", "#E45756"


def main():
    FIGURES.mkdir(exist_ok=True)
    nodes = load_nodes()
    train = nodes[nodes["split"] == "train"]
    lab = train[train["label"] >= 0]
    licit, illicit = lab[lab["label"] == 0], lab[lab["label"] == 1]
    feats = LOCAL + AGG + GRAPH + HOP2
    out = {}

    # Per-step label composition (train steps only).
    per_step = lab.groupby("time_step")["label"].agg(["sum", "count"])
    out["train_illicit_ratio_per_step"] = (per_step["sum"] / per_step["count"]).round(4).to_dict()

    # Univariate separability: KS distance and mutual information.
    ks = {f: ks_2samp(licit[f], illicit[f]).statistic for f in feats}
    mi = dict(zip(feats, mutual_info_classif(lab[feats], lab["label"], random_state=0)))
    top = sorted(ks, key=ks.get, reverse=True)[:10]
    out["top10_ks"] = [{"feature": f, "ks": round(ks[f], 3), "mi": round(mi[f], 3),
                        "median_licit": round(float(licit[f].median()), 3),
                        "median_illicit": round(float(illicit[f].median()), 3)} for f in top]
    groups = {"local": LOCAL, "agg": AGG, "degree": GRAPH, "hop2": HOP2}
    out["mean_ks"] = {g: float(np.mean([ks[f] for f in cols])) for g, cols in groups.items()}
    out["mean_mi"] = {g: float(np.mean([mi[f] for f in cols])) for g, cols in groups.items()}
    out["n_features_ks_gt_0.3"] = {g: int(sum(ks[f] > 0.3 for f in cols)) for g, cols in groups.items()}

    # Redundancy among features.
    corr = np.corrcoef(lab[LOCAL + AGG].to_numpy().T)
    iu = np.triu_indices_from(corr, k=1)
    out["pairs_abs_corr_gt_0.95"] = int(np.sum(np.abs(corr[iu]) > 0.95))

    # 3-sigma view: share of rows with at least one |z| > 3 w.r.t. licit statistics.
    mu, sd = licit[LOCAL].mean(), licit[LOCAL].std() + 1e-9
    z_licit = ((licit[LOCAL] - mu) / sd).abs().max(axis=1)
    z_illicit = ((illicit[LOCAL] - mu) / sd).abs().max(axis=1)
    out["share_maxz_gt_3"] = {"licit": float((z_licit > 3).mean()),
                              "illicit": float((z_illicit > 3).mean())}

    # Temporal drift of licit behaviour inside train: early (1-14) vs late (15-29) steps.
    early = licit[licit["time_step"] <= 14]
    late = licit[licit["time_step"] > 14]
    drift = [ks_2samp(early[f], late[f]).statistic for f in LOCAL]
    out["licit_drift_ks_early_vs_late"] = {"mean": float(np.mean(drift)),
                                           "n_gt_0.2": int(np.sum(np.array(drift) > 0.2))}
    ill_early = illicit[illicit["time_step"] <= 14]
    ill_late = illicit[illicit["time_step"] > 14]
    drift_ill = [ks_2samp(ill_early[f], ill_late[f]).statistic for f in LOCAL]
    out["illicit_drift_ks_early_vs_late"] = {"mean": float(np.mean(drift_ill)),
                                             "n_gt_0.2": int(np.sum(np.array(drift_ill) > 0.2))}

    out["degree"] = {name: {g: float(df[g].mean()) for g in GRAPH}
                     for name, df in [("licit", licit), ("illicit", illicit)]}
    save_json(out, "eda.json")

    # Figure: (a) top-KS feature histogram, (b) CDF of max |z|, (c) PCA projection.
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    ax = axes[0]
    f = top[0]
    lo, hi = np.percentile(lab[f], [1, 99])
    bins = np.linspace(lo, hi, 40)
    ax.hist(licit[f].clip(lo, hi), bins=bins, density=True, alpha=0.6, color=LICIT_C, label="licit")
    ax.hist(illicit[f].clip(lo, hi), bins=bins, density=True, alpha=0.6, color=ILLICIT_C, label="illicit")
    ax.set_xlabel(f"{f} (standardized units)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title(f"(a) top feature, KS={ks[f]:.2f}")

    ax = axes[1]
    for z, name, color in [(z_licit, "licit", LICIT_C), (z_illicit, "illicit", ILLICIT_C)]:
        v = np.sort(z.to_numpy())
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=color, label=name)
    ax.axvline(3, color="gray", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel(r"max$_j |z_j|$ (std. units, log)")
    ax.set_ylabel("cumulative share")
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(r"(b) 3$\sigma$ outlyingness")

    ax = axes[2]
    pca = PCA(n_components=2, random_state=0).fit(licit[LOCAL + AGG])
    rng = np.random.default_rng(0)
    sub = licit.iloc[rng.choice(len(licit), 3000, replace=False)]
    pl, pi = pca.transform(sub[LOCAL + AGG]), pca.transform(illicit[LOCAL + AGG])
    ax.scatter(pl[:, 0], pl[:, 1], s=2, alpha=0.4, color=LICIT_C, label="licit (3k sample)")
    ax.scatter(pi[:, 0], pi[:, 1], s=2, alpha=0.4, color=ILLICIT_C, label="illicit")
    ax.set_xlim(*np.percentile(np.r_[pl[:, 0], pi[:, 0]], [0.5, 99.5]))
    ax.set_ylim(*np.percentile(np.r_[pl[:, 1], pi[:, 1]], [0.5, 99.5]))
    ax.set_xlabel("PC1 (std. units, fit on licit)")
    ax.set_ylabel("PC2 (std. units)")
    ax.set_title("(c) PCA projection")
    fig.tight_layout()
    fig.savefig(FIGURES / "eda.pdf")

    print(out)


if __name__ == "__main__":
    main()
