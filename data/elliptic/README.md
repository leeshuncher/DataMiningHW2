# Elliptic Data Set (unmodified copy)

The original archive from the official Kaggle release
(<https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>). It is split into
40 MB parts only to stay under GitHub's 100 MB file-size limit; the bytes are unchanged.

- **Source:** Elliptic, released with Weber et al. (2019), *Anti-Money Laundering in Bitcoin:
  Experimenting with Graph Convolutional Networks for Financial Forensics*, KDD '19 Workshop on
  Anomaly Detection in Finance.
- **License:** [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/).
  Redistributed here unmodified and for non-commercial coursework.
- **Integrity:** the reassembled `elliptic.zip` has SHA-256
  `70ed052c52002392720615a03261039b62d1ad594f6081ad021cb174a8ad5717`.
  Each part is listed in `SHA256SUMS`.

`python scripts/download_data.py` reassembles the parts, verifies the checksum and extracts
the three CSV files into `data/raw/`:

- `elliptic_txs_features.csv`
- `elliptic_txs_classes.csv`
- `elliptic_txs_edgelist.csv`

To do it by hand:

```bash
cat data/elliptic/elliptic.zip.part-* > elliptic.zip && sha256sum elliptic.zip
```
