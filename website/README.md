# FinInteract project page

Public project page for **FinInteract: Benchmarking Clarification and Intent Integration in Ambiguous Financial Question Answering**, accepted to Findings of ACL: EACL 2027.

Authors: Xinyu Wang, Tung Sum Thomas Kwok, Zhenghan Tai, Guang Cheng. Correspondence: guangcheng@ucla.edu.

This is a buildless static website. Serve the directory with `python -m http.server 8000` and visit `http://localhost:8000`. GitHub Pages serves the default branch root; `.nojekyll` preserves assets directly.

The site includes the author manuscript and two clearly distinguished dataset versions:

- **v1.0, 173 items:** frozen snapshot used for the reported experiments.
- **v1.1, 172 items:** corrected release, excluding `fininteract_0172` after human adjudication. Recompute scores when changing versions.

See `assets/data/dataset-manifest.json` for hashes and counts and `assets/data/verified-results.json` for result denominators and limitations. Research code: https://github.com/ThomasK1018/fininteract.

Code is MIT licensed. Original benchmark records are CC BY 4.0; quoted source filings and third-party materials retain their own rights and terms. See `LICENSE` and `DATA_LICENSE.txt`.
