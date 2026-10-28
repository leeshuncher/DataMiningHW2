"""Graph neural network detectors (PyTorch Geometric) on the full transaction graph.

The graph is used full-batch: Elliptic edges never cross time steps, so the
representation of a training node never depends on validation or test nodes.
Loss is computed on training nodes only; validation PR-AUC picks the epoch
(evaluated every 10 epochs, stop after 100 epochs without improvement).

Two roles share one model:
  "sup": licit (0) vs illicit (1) labels   -> supervised Tier 3 reference
  "pu":  licit (0) vs unknown (1) targets  -> graph version of our LvU detector
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import to_undirected

from common import PROCESSED

torch.set_num_threads(4)
CONVS = {"gcn": GCNConv, "sage": SAGEConv}


def edge_index(nodes):
    """Undirected edge index in row order of `nodes`."""
    edges = pd.read_parquet(PROCESSED / "edges.parquet")
    index = pd.Series(np.arange(len(nodes)), index=nodes["txId"])
    src, dst = index[edges["src"]].to_numpy(), index[edges["dst"]].to_numpy()
    return to_undirected(torch.tensor(np.vstack([src, dst]), dtype=torch.long))


class Net(torch.nn.Module):
    """Two graph-convolution layers with ReLU and dropout; one logit per node."""

    def __init__(self, conv, in_dim, hidden, dropout=0.3):
        super().__init__()
        self.c1 = CONVS[conv](in_dim, hidden)
        self.c2 = CONVS[conv](hidden, 1)
        self.dropout = dropout

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.c1(x, ei)), self.dropout, self.training)
        return self.c2(h, ei).squeeze(-1)


class GNNDetector:
    """Fits on the train split and returns scores for every node (higher = more anomalous)."""

    def __init__(self, role, conv="sage", hidden=64, epochs=1000, patience=100, lr=0.01, seed=0):
        self.role, self.conv, self.hidden, self.seed = role, conv, hidden, seed
        self.epochs, self.patience, self.lr = epochs, patience, lr

    def fit_score(self, nodes, ei, cols):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        split, y = nodes["split"].to_numpy(), nodes["label"].to_numpy()
        train = split == "train"
        X = nodes[cols].to_numpy(np.float32)
        mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-6  # scaler fit on train only
        x = torch.tensor((X - mu) / sd)

        if self.role == "pu":
            mask, target = train & (y != 1), (y == -1)  # illicit training rows are not used
        else:
            mask, target = train & (y >= 0), (y == 1)
        target = torch.tensor(target.astype(np.float32))
        mask_t = torch.tensor(mask)
        pos_weight = torch.tensor((target[mask_t] == 0).sum().item() / max((target[mask_t] == 1).sum().item(), 1))

        val = (split == "val") & (y >= 0)
        net = Net(self.conv, x.shape[1], self.hidden)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=5e-4)
        best_ap, best_scores, self.best_epoch = -1.0, None, 0
        start = time.perf_counter()
        for epoch in range(1, self.epochs + 1):
            net.train()
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(net(x, ei)[mask_t], target[mask_t], pos_weight=pos_weight)
            loss.backward()
            opt.step()
            if epoch % 10 == 0:  # early stopping on validation PR-AUC
                net.eval()
                with torch.no_grad():
                    s = torch.sigmoid(net(x, ei)).numpy()
                ap = average_precision_score(y[val], s[val])
                if ap > best_ap:
                    best_ap, best_scores, self.best_epoch = ap, s, epoch
                elif epoch - self.best_epoch >= self.patience:
                    break
        self.fit_s = time.perf_counter() - start
        self.val_pr_auc = float(best_ap)

        # Inference cost: one full-graph forward pass, per node.
        net.eval()
        start = time.perf_counter()
        with torch.no_grad():
            net(x, ei)
        self.ms_per_sample = 1000 * (time.perf_counter() - start) / len(nodes)
        return best_scores
