"""Generate the LaTeX tables of the paper directly from results/*.json.

Usage: python paper/make_tables.py   (writes paper/tables/*.tex)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "paper" / "tables"

OURS = "T2 Licit-vs-Unlabeled (ours)"
DISPLAY = {
    "T1a random": ("T1a", "Random score"),
    "T1b 3-sigma": ("T1b", r"3$\sigma$ rule (max $|z|$)"),
    "T2' IForest (one-class)": ("T2$'$", "One-class Isolation Forest"),
    OURS: ("T2", r"\textbf{LvU classifier (ours)}"),
    "T3a RandomForest (sup.)": ("T3a", "Random Forest (sup.)"),
    "T3b XGBoost (sup.)": ("T3b", "XGBoost (sup.)"),
}


def load(name):
    return json.loads((RES / name).read_text())


def pm(m, digits=3):
    return f"{m['mean']:.{digits}f}" + (rf"$\pm${m['std']:.{digits}f}" if m["std"] > 0 else "")


def write(name, body):
    OUT.mkdir(exist_ok=True)
    (OUT / name).write_text(body)


def data_table():
    st = json.loads((ROOT / "data" / "processed" / "stats.json").read_text())
    rows = []
    for split, s in st["splits"].items():
        lo, hi = s["time_steps"]
        rows.append(f"{split} & {lo}--{hi} & {s['transactions']:,} & {s['licit']:,} & {s['illicit']:,} & "
                    f"{s['unknown']:,} & {100 * s['illicit_ratio_of_labeled']:.1f}\\% \\\\")
    o = st["overall"]
    rows.append(r"\midrule")
    rows.append(f"all & 1--49 & {o['transactions']:,} & {o['licit']:,} & {o['illicit']:,} & {o['unknown']:,} & "
                f"{100 * o['illicit_ratio_of_labeled']:.1f}\\% \\\\")
    write("data.tex", "\n".join([
        r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
        r"Split & Steps & Tx & Licit & Illicit & Unknown & Ill./lab. \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


def main_table():
    main, ana = load("main.json"), load("analysis.json")
    cols = ["pr_auc", "roc_auc", "f1", "p_at_k"]
    best = {c: max(main[n]["summary"][c]["mean"] for n in main) for c in cols}
    pre = ana["pr_auc_shutdown"]
    best_pre = max(v["before"] for v in pre.values())
    rows = []
    for name, (tier, label) in DISPLAY.items():
        s = main[name]["summary"]
        cells = []
        for c in cols:
            v = pm(s[c])
            cells.append(rf"\textbf{{{v}}}" if abs(s[c]["mean"] - best[c]) < 1e-9 else v)
        p = f"{pre[name]['before']:.3f}"
        cells.append(rf"\textbf{{{p}}}" if abs(pre[name]["before"] - best_pre) < 1e-9 else p)
        cells.append(f"{pre[name]['from']:.3f}")
        cells.append(f"{s['fit_s']['mean']:.1f}")
        cells.append(f"{1000 * s['ms_per_sample']['mean']:.1f}")
        rows.append(f"{tier} & {label} & " + " & ".join(cells) + r" \\")
        if tier in ("T1b", "T2"):
            rows.append(r"\midrule")
    prev = pre[OURS]
    write("main.tex", "\n".join([
        r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
        r" & & & & & & \multicolumn{2}{c}{PR-AUC by period} & Fit & Score \\",
        r"\cmidrule(lr){7-8}",
        r"Tier & Method & PR-AUC & ROC-AUC & F1@$\tau$ & P@100 & steps 35--42 & 43--49 & (s) & ($\mu$s/tx) \\",
        r"\midrule", *rows,
        rf"\multicolumn{{2}}{{l}}{{\emph{{prevalence (= random PR-AUC)}}}} & 0.065 & & & & "
        rf"{prev['prevalence_before']:.3f} & {prev['prevalence_from']:.3f} & & \\",
        r"\bottomrule", r"\end{tabular}", ""]))


def ablation_table():
    ab = load("ablations.json")
    ref = ab["A1_contamination"]["0%"]["pr_auc"]["mean"]
    groups = [
        ("A1: illicit rows hidden in licit side", "A1_contamination",
         {"0%": "0\\% (ours)", "1%": "1\\%", "5%": "5\\%", "10%": "10\\%"}),
        ("A2: fraction of unlabeled pool used", "A2_unlabeled_pool",
         {"5%": "5\\%", "25%": "25\\%", "50%": "50\\%", "100%": "100\\% (ours)"}),
        ("A3: label-free scoring function", "A3_scoring",
         {"One-class IForest": "one-class IF", "One-class Mahalanobis": "one-class Mahalanobis",
          "One-class LOF": "one-class LOF", "PU kNN density ratio": "PU $k$NN ratio",
          "PU classifier (ours)": "PU classifier (ours)"}),
        ("A4: feature set", "A4_features",
         {"LF": "LF, 93 local (ours)", "AF": "AF (+72 aggregates)", "AF+deg": "AF + in/out degree"}),
    ]
    rows = []
    for title, key, labels in groups:
        rows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for k, label in labels.items():
            m = ab[key][k]
            d = m["pr_auc"]["mean"] - ref
            delta = "---" if "(ours)" in label else f"${d:+.3f}$"
            rows.append(f"\\quad {label} & {pm(m['pr_auc'])} & {delta} & {m['f1']['mean']:.3f} \\\\")
    write("ablation.tex", "\n".join([
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Setting & PR-AUC & $\Delta$ & F1@$\tau$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


def threshold_table():
    ab = load("ablations.json")["A5_threshold"]
    labels = {"best_f1": r"best F1 on val (ours)", "licit_p95": r"95th pct.\ licit train",
              "weekly_budget": r"top-30 per time step"}
    rows = [f"{labels[k]} & {m['alerts']['mean']:.0f} & {m['precision']['mean']:.3f} & {m['recall']['mean']:.3f} & {m['f1']['mean']:.3f} \\\\"
            for k, m in ab.items()]
    write("threshold.tex", "\n".join([
        r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
        r"$\tau$ rule & Alerts & Precision & Recall & F1 \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


CAUSES = {
    "FN (inversion)": "licit-typical profile",
    "FN (near miss)": "73/93 features equal \\#4",
    "FP (confident)": "licit tx in illicit mode",
    "FP (near miss)": "73/93 features equal \\#2",
}


def cases_table():
    ea = load("analysis.json")["error_analysis"]
    rows = []
    for i, c in enumerate([c for c in ea["cases"] if c["type"] in CAUSES], 1):
        f = c["top_features"][0]
        rows.append(f"{i} & {c['type'].split()[0]} & {c['txId']} & {c['time_step']} & {c['score']:.3f} & "
                    f"{f['feature'].replace('_', chr(92) + '_')}$={f['value']:.2f}$ & {c['rf_score']:.2f} & "
                    f"{CAUSES[c['type']]} \\\\")
    write("cases.tex", "\n".join([
        r"\begin{tabular}{@{}clrrcccl@{}}", r"\toprule",
        r"\# & Type & txId & Step & $s(x)$ & Top-SHAP feature & RF & Likely cause \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


if __name__ == "__main__":
    data_table()
    main_table()
    ablation_table()
    threshold_table()
    cases_table()
