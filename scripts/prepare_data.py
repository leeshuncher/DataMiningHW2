"""Turn the raw Elliptic CSVs into tidy parquet files with one shared split.

Outputs (data/processed/):
    nodes.parquet  one row per transaction: txId, time_step, label, split,
                   local_1..local_93, agg_1..agg_72
    edges.parquet  directed payment-flow edges: src, dst (txIds)
    stats.json     dataset summary used in the paper (§3)

Labels: 1 = illicit (anomaly), 0 = licit (normal), -1 = unknown (unlabeled).

Split (chronological, by time step; identical for every method):
    train  steps  1-29   one-class models fit on licit rows only
    val    steps 30-34   threshold / hyperparameter tuning only
    test   steps 35-49   evaluated once
The train+val / test boundary follows the 70/30 temporal split of Weber et al. (2019).
"""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "elliptic_bitcoin_dataset"
OUT_DIR = ROOT / "data" / "processed"

N_LOCAL = 93  # local features besides the time step
N_AGG = 72  # features aggregated over one-hop neighbours
SPLITS = {"train": (1, 29), "val": (30, 34), "test": (35, 49)}
LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}


def load_nodes():
    columns = (
        ["txId", "time_step"]
        + [f"local_{i}" for i in range(1, N_LOCAL + 1)]
        + [f"agg_{i}" for i in range(1, N_AGG + 1)]
    )
    feats = pd.read_csv(RAW_DIR / "elliptic_txs_features.csv", header=None, names=columns)
    feats[columns[2:]] = feats[columns[2:]].astype("float32")

    classes = pd.read_csv(RAW_DIR / "elliptic_txs_classes.csv", dtype={"class": str})
    classes["label"] = classes["class"].map(LABEL_MAP).astype("int8")

    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left", validate="1:1").copy()
    assert nodes["label"].notna().all(), "transactions without a class row"

    nodes["split"] = ""
    for name, (lo, hi) in SPLITS.items():
        nodes.loc[nodes["time_step"].between(lo, hi), "split"] = name
    assert (nodes["split"] != "").all(), "time steps outside the defined splits"

    front = ["txId", "time_step", "label", "split"]
    return nodes[front + columns[2:]]


def load_edges(tx_ids):
    edges = pd.read_csv(RAW_DIR / "elliptic_txs_edgelist.csv")
    edges.columns = ["src", "dst"]
    assert edges["src"].isin(tx_ids).all() and edges["dst"].isin(tx_ids).all()
    return edges


def summarize(nodes, edges):
    def counts(df):
        c = df["label"].value_counts()
        n_illicit, n_licit = int(c.get(1, 0)), int(c.get(0, 0))
        labeled = n_illicit + n_licit
        return {
            "transactions": len(df),
            "illicit": n_illicit,
            "licit": n_licit,
            "unknown": int(c.get(-1, 0)),
            "illicit_ratio_of_labeled": round(n_illicit / labeled, 4) if labeled else None,
        }

    return {
        "n_features": N_LOCAL + N_AGG + 1,
        "n_edges": len(edges),
        "time_steps": [int(nodes["time_step"].min()), int(nodes["time_step"].max())],
        "splits": {name: {"time_steps": list(SPLITS[name]), **counts(nodes[nodes["split"] == name])}
                   for name in SPLITS},
        "overall": counts(nodes),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes = load_nodes()
    edges = load_edges(set(nodes["txId"]))

    nodes.to_parquet(OUT_DIR / "nodes.parquet", index=False)
    edges.to_parquet(OUT_DIR / "edges.parquet", index=False)

    stats = summarize(nodes, edges)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
