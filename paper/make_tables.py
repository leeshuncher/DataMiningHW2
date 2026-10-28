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
    """mean +- seed std; deterministic results show a bootstrap std marked with a dagger."""
    if "boot_std" in m:
        return f"{m['mean']:.{digits}f}" + rf"$\pm${m['boot_std']:.{digits}f}$^\dagger$"
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


CONV_NAME = {"gcn": "GCN", "sage": "GraphSAGE"}


def main_table():
    main, ana, gnn = load("main.json"), load("analysis.json"), load("gnn.json")
    pre = ana["pr_auc_shutdown"]
    for name, res in main.items():
        if len(res["runs"]) == 1:  # deterministic: report the test-bootstrap std
            res["summary"]["pr_auc"]["boot_std"] = ana["bootstrap_std"][name]
    # (tier, label, summary, PR-AUC before / from the shutdown), in table order; None = rule.
    entries = []
    for name, (tier, label) in DISPLAY.items():
        if tier == "T2":
            g = gnn["T2'' LvU-GNN"]
            entries.append(("T2$''$", f"LvU-{CONV_NAME[g['config']['conv']]}", g["summary"],
                            g["summary"]["pr_auc_before"], g["summary"]["pr_auc_from"]))
        entries.append((tier, label, main[name]["summary"], pre[name]["before"], pre[name]["from"]))
        if tier in ("T1b", "T2"):
            entries.append(None)
    g = gnn["T3c GNN (sup.)"]
    entries.append(("T3c", f"{CONV_NAME[g['config']['conv']]} (sup.)", g["summary"],
                    g["summary"]["pr_auc_before"], g["summary"]["pr_auc_from"]))

    cols = ["pr_auc", "roc_auc", "f1", "p_at_k"]
    rows_ = [e for e in entries if e]
    best = {c: max(e[2][c]["mean"] for e in rows_) for c in cols}
    best_pre = max(e[3] for e in rows_)
    rows = []
    for e in entries:
        if e is None:
            rows.append(r"\midrule")
            continue
        tier, label, s, before, after = e
        cells = []
        for c in cols:
            v = pm(s[c])
            cells.append(rf"\textbf{{{v}}}" if abs(s[c]["mean"] - best[c]) < 1e-9 else v)
        p = f"{before:.3f}"
        cells.append(rf"\textbf{{{p}}}" if abs(before - best_pre) < 1e-9 else p)
        cells.append(f"{after:.3f}")
        fit = s["fit_s"]["mean"]
        cells.append(f"{fit:.0f}" if fit >= 10 else f"{fit:.1f}")
        cells.append(f"{1000 * s['ms_per_sample']['mean']:.1f}")
        rows.append(f"{tier} & {label} & " + " & ".join(cells) + r" \\")
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
         {"LF": "LF, 93 local (ours)", "AF": "AF (+72 aggregates)", "AF+deg": "AF + in/out degree", "AF+2hop": "AF + 2-hop aggregates"}),
    ]
    rows = []
    for title, key, labels in groups:
        rows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{title}}}}} \\")
        for k, label in labels.items():
            m = ab[key][k]
            d = m["pr_auc"]["mean"] - ref
            delta = "---" if "(ours)" in label else f"${d:+.3f}$"
            rows.append(f"\\quad {label} & {pm(m['pr_auc'])} & {delta} & {m['val_pr_auc']['mean']:.3f} \\\\")
    write("ablation.tex", "\n".join([
        r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
        r"Setting & Test PR-AUC & $\Delta$ & Val \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


CASES = ["FN (inversion)", "FN (near miss)", "FP (confident)", "FP (near miss)"]


def signed(v):
    """Signed margin; tiny values in scientific notation so that the sign stays visible."""
    if abs(v) >= 1e-3:
        return f"${v:+.3f}$"
    mant, exp = f"{v:+.1e}".split("e")
    return rf"${mant}{{\times}}10^{{{int(exp)}}}$"


def cases_table():
    ea = load("analysis.json")["error_analysis"]
    rows = []
    for i, c in enumerate([c for c in ea["cases"] if c["type"] in CASES], 1):
        f = c["top_features"][0]
        rows.append(f"{i} & {c['type'].split()[0]} & {c['txId']} & {c['time_step']} & {c['score']:.5f} & "
                    f"{signed(c['margin'])} & {f['feature'].replace('local_', 'L')}$={f['value']:.2f}$ "
                    f"(${f['shap']:+.2f}$) & {c['rf_score']:.2f} \\\\")
    write("cases.tex", "\n".join([
        r"\begin{tabular}{@{}clrrcrcr@{}}", r"\toprule",
        r"\# & Type & txId & Step & $s(x)$ & $s-\tau$ & Top SHAP & RF \\",
        r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


if __name__ == "__main__":
    data_table()
    main_table()
    ablation_table()
    cases_table()
