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
    "T1b 3-sigma": ("T1b", r"3$\sigma$ rule"),
    "T2' IForest (one-class)": ("T2$'$", "One-class IF"),
    OURS: ("T2", r"\textbf{LvU (ours)}"),
    "T3a RandomForest (sup.)": ("T3a", "RF (sup.)"),
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


SHUTDOWN_STEP = 43


def period_f1(runs, n_illicit):
    """Pooled F1@tau before / from the shutdown, from per-step alerts and recall (mean over seeds)."""
    out = {"before": [], "from": []}
    for r in runs:
        for period, keep in [("before", lambda t: t < SHUTDOWN_STEP), ("from", lambda t: t >= SHUTDOWN_STEP)]:
            steps = [t for t in r["per_step"] if keep(int(t))]
            tp = sum(r["per_step"][t]["recall"] * n_illicit[t] for t in steps)
            alerts = sum(r["per_step"][t]["alerts"] for t in steps)
            pos = sum(n_illicit[t] for t in steps)
            out[period].append(2 * tp / (alerts + pos))
    return {k: sum(v) / len(v) for k, v in out.items()}


def main_table():
    main, ana, gnn = load("main.json"), load("analysis.json"), load("gnn.json")
    pre = ana["pr_auc_shutdown"]
    n_illicit = {t: v["n_illicit"] for t, v in main[OURS]["runs"][0]["per_step"].items()}
    for name, res in main.items():
        if len(res["runs"]) == 1:  # deterministic: report the test-bootstrap std of every metric
            for metric, sd in ana["bootstrap_std_deterministic"][name].items():
                res["summary"][metric]["boot_std"] = sd
    # (tier, label, summary, PR-AUC before / from the shutdown, runs), in table order; None = rule.
    entries = []
    for name, (tier, label) in DISPLAY.items():
        if tier == "T2":
            g = gnn["T2'' LvU-GNN"]
            entries.append(("T2$''$", f"LvU-{CONV_NAME[g['config']['conv']]}", g["summary"],
                            g["summary"]["pr_auc_before"], g["summary"]["pr_auc_from"], g["runs"]))
        entries.append((tier, label, main[name]["summary"], pre[name]["before"], pre[name]["from"],
                         main[name]["runs"]))
        if tier in ("T1b", "T2"):
            entries.append(None)
    g = gnn["T3c GNN (sup.)"]
    entries.append(("T3c", f"{CONV_NAME[g['config']['conv']]} (sup.)", g["summary"],
                    g["summary"]["pr_auc_before"], g["summary"]["pr_auc_from"], g["runs"]))

    cols = ["pr_auc", "roc_auc", "f1", "p_at_k"]
    rows_ = [e for e in entries if e]
    best = {c: max(e[2][c]["mean"] for e in rows_) for c in cols}
    best_pre = max(e[3] for e in rows_)
    f1p = {e[1]: period_f1(e[5], n_illicit) for e in rows_}
    best_f1_pre = max(v["before"] for v in f1p.values())
    rows = []
    for e in entries:
        if e is None:
            rows.append(r"\midrule")
            continue
        tier, label, s, before, after, _ = e
        cells = []
        for c in cols:
            v = pm(s[c])
            cells.append(rf"\textbf{{{v}}}" if abs(s[c]["mean"] - best[c]) < 1e-9 else v)
        p = f"{before:.3f}"
        cells.append(rf"\textbf{{{p}}}" if abs(before - best_pre) < 1e-9 else p)
        cells.append(f"{after:.3f}")
        f = f"{f1p[label]['before']:.2f}"
        cells.append(rf"\textbf{{{f}}}" if abs(f1p[label]["before"] - best_f1_pre) < 1e-9 else f)
        cells.append(f"{f1p[label]['from']:.2f}")
        fit = s["fit_s"]["mean"]
        cells.append(f"{fit:.0f}" if fit >= 10 else f"{fit:.1f}")
        cells.append(f"{1000 * s['ms_per_sample']['mean']:.1f}")
        rows.append(f"{tier} & {label} & " + " & ".join(cells) + r" \\")
    prev = pre[OURS]
    write("main.tex", "\n".join([
        r"\begin{tabular}{@{}llcccccccccc@{}}", r"\toprule",
        r" & & & & & & \multicolumn{2}{c}{PR-AUC, steps} & \multicolumn{2}{c}{F1, steps} & Fit & Score \\",
        r"\cmidrule(lr){7-8}\cmidrule(lr){9-10}",
        r"Tier & Method & PR-AUC & ROC-AUC & F1@$\tau$ & P@100 & 35--42 & 43--49 & 35--42 & 43--49 & (s) & ($\mu$s/tx) \\",
        r"\midrule", *rows,
        rf"\multicolumn{{2}}{{l}}{{\emph{{prevalence}}}} & 0.065 & & & & "
        rf"{prev['prevalence_before']:.3f} & {prev['prevalence_from']:.3f} & & & & \\",
        r"\bottomrule", r"\end{tabular}", ""]))


def threshold_table():
    ab = load("ablations.json")["A5_threshold"]
    tests = load("analysis.json")["threshold_rule_tests"]
    labels = {"best_f1": r"$\tau_{\mathrm{F1}}$", "cost_1": r"$\tau_1$", "cost_5": r"$\tau_5$",
              "cost_10": r"$\tau_{10}$", "cost_20": r"$\tau_{20}$",
              "licit_p95": r"licit 95th pct.", "weekly_budget": r"top 30/step"}
    rows = []
    for k, label in labels.items():
        m = ab[k]
        prec = m["precision"]["mean"]
        if k == "best_f1":
            p = "---"
        else:
            pv = tests[k]["f1"]["p"]
            p = "$<$0.001" if pv < 1e-3 else f"{pv:.2f}"
        rows.append(f"{label} & {m['alerts']['mean']:,.0f} & {prec:.2f} & {m['recall']['mean']:.2f} & "
                    f"{m['f1']['mean']:.2f} & {(1 - prec) / prec:.1f} & {p} \\\\")
    write("threshold.tex", "\n".join([
        r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
        r"Alert rule & Alerts & Prec. & Rec. & F1 & FP/catch & $p$ \\", r"\midrule",
        *rows, r"\bottomrule", r"\end{tabular}", ""]))


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
    threshold_table()
    cases_table()
