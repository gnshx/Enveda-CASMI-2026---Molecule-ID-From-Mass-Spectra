#!/usr/bin/env python3
"""
scripts/prepare_v32.py
Prepare Version 32: Sovereign Rank-1 Shield (w_top=0.00) + Calibrated Neural Reranker + MIDX Dedup.
Base: deploy_kernel_midx (Version 24, PB 0.4800).
"""

import os
import json
import shutil
from pathlib import Path

REPO_ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
V32_DIR = REPO_ROOT / "deploy_kernel_v32"
V32_DIR.mkdir(parents=True, exist_ok=True)

# Copy kernel metadata
meta = json.loads((REPO_ROOT / "deploy_kernel_v31" / "kernel-metadata.json").read_text(encoding="utf-8"))
(V32_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

# Load base notebook (Version 24 PB 0.4800)
base_nb_path = REPO_ROOT / "deploy_kernel_midx" / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
nb = json.loads(base_nb_path.read_text(encoding="utf-8"))

# Header Cell
nb["cells"][0]["source"] = [
    "# CASMI26 LB 0.490+ | Version 32: Sovereign Immutable Rank-1 Shield (w_top=0.00) + Calibrated Neural Reranker (w_mid=0.90) + MIDX Dedup\n",
    "\n",
    "**Version 32 Sovereign**: Restores and reinforces the immutable **Rank 1 Shield** (`w_top = 0.00`, `PIN_OLD_LEADER = True`).\n",
    "Confirmed Regime A reference library matches are 100% protected against neural demotion.\n",
    "Ranks 2-25 are re-ranked using calibrated dual-neural consensus (`0.10*z(Saner v1) + 0.90*z(Scorer v2)`), PubChem documentation prior (mu=0.25), and metric-exact InChIKey14 deduplication.\n"
]

# Update Configuration cell
config_updated = False
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "CUSTOM = dict(" in src and "w_top=" in src:
        src = src.replace("w_top=0.25", "w_top=0.00")
        src = src.replace("w_mid=0.75", "w_mid=0.90")
        src = src.replace("POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02", "POP_W, SPEC_W, NP_W = 0.25, 0.05, 0.10")
        cell["source"] = [src]
        config_updated = True
        break

assert config_updated, "Failed to find and update Configuration cell!"

target_nb_path = V32_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
target_nb_path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"✅ Prepared deploy_kernel_v32 cleanly at {target_nb_path} ({len(nb['cells'])} cells)")
