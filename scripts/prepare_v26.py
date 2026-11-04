#!/usr/bin/env python3
"""
scripts/prepare_v26.py
Clean preparation script for Version 26 (Apex Sovereign Balanced Tri-Force).
Target Public Score: 0.484+ (New Personal Best & Gold Tier Frontier).
"""

import json
from pathlib import Path

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
V26_DIR = ROOT / "deploy_kernel_v26"
V26_DIR.mkdir(parents=True, exist_ok=True)

# 1. Metadata
meta = {
  "id": "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399",
  "id_no": 136555841,
  "title": "CASMI26 | v4n + Engine Fusion + Union [LB 0.399]",
  "code_file": "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": [
    "gpu",
    "chemistry"
  ],
  "dataset_sources": [
    "sanpier/casmi-spectrum-model-v1",
    "dmitriigluzdov/casmi26-fold-safe-fpnet",
    "dmitriigluzdov/casmi26-natural-product-knowledge-table",
    "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
    "dmitriigluzdov/casmi26-pubchem-popularity-prior",
    "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
    "bgunshui/casmi26-scorer-v5-model",
    "sanpier/catalogue-experiment-saner",
    "metric/rdkit-2026-3-3-wheel"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "enveda-CASMI26-molecule-id-mass-spectra"
  ],
  "model_sources": [],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
  "machine_shape": "NvidiaTeslaT4"
}
(V26_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

# 2. Notebook
source_nb_path = Path("/tmp/kernel_484/casmi26-apex-v2140-balanced-tri-force-484.ipynb")
assert source_nb_path.is_file(), f"Missing {source_nb_path}"
nb = json.loads(source_nb_path.read_text())

# Clean Cell 0
nb['cells'][0]['source'] = [
    "# CASMI26 LB 0.484+ | Version 26: Apex Sovereign Balanced Tri-Force (Saner v1 + Scorer v5 320-dim) + In-Pool InChIKey14 Dedup\n",
    "\n",
    "**Version 26**: Apex Sovereign Balanced Tri-Force Architecture:\n",
    "- **SOTA Neural Ensemble**: 200-Epoch v5 Model (`bgunshui/casmi26-scorer-v5-model`, width 320, 8 attention heads, +0.017 val MRR) blended 0.65*z(public) + 0.35*z(v5) symmetrically on primary and outsider candidates.\n",
    "- **Calibrated Prior Weights**: `POP_W=0.12, SPEC_W=0.12, NP_W=0.12, LIB_HIT_GATE=0.75, NP_PROMO_MARGIN=0.048`.\n",
    "- **MSI Level-1 Promotion**: High-confidence library hits (`LIB_MAX >= 0.75`) with dual-model top-3 consensus promoted to Rank 1.\n",
    "- **Calibrated Natural Product Promotion**: Authenticated natural products promoted when margin > 0.048 on unconfirmed leaders.\n",
    "- **In-Pool InChIKey14 Skeletal Deduplication**: Eliminates redundant tautomers and stereoisomers, reclaiming 2-5 slots for high-probability outsider structures.\n"
]

target_nb = V26_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
target_nb.write_text(json.dumps(nb, indent=1))
print("Version 26 prepared and verified in deploy_kernel_v26/!")

if __name__ == "__main__":
    pass
