# Roman Urdu Normalizer — dataset

A dataset for normalizing noisy Roman Urdu (chat spelling, typos, stretched words) to one standard spelling,
e.g. `mujhy kl ghr jana h` → `mujhe kal ghar jana hai`.

- **Generated data:** [`dataset/data/v4/`](dataset/data/v4/) — start with `golden_pairs.csv` (68,166 noisy → normalized
  pairs) and read [`dataset/data/v4/README.md`](dataset/data/v4/README.md) for splits, conventions and known limits.
- **Inputs:** [`external_raw/`](external_raw/) (`dataset.csv`, UCI `Roman Urdu DataSet.csv`) and `dataset/sources/`.
- **Rebuild:** `cd dataset && python build_v4.py build && python audit_v4.py` (Python 3, standard library only).
- **Docs:** [`docs/`](docs/) — project plan PDF and methodology diagram.

| Split | Sentences |
|---|---|
| train | 48,703 |
| dev | 4,831 |
| test_social (real comments, reviewed) | 297 |
| test_social_uci | 186 |
| test_real (Dakshina) | 394 |
| test_synth_iid | 2,000 |
| test_synth_ood | 11,755 |

The real-comment test gold was reviewed by an AI assistant and has not yet been checked by a human annotator.

## Project members

- Abrar Ali
- Sharjeel Anjum
- Mian Ahmad Mehmood
- Toqir Dar
