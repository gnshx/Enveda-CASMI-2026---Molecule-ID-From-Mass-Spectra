#!/usr/bin/env python3
import json
from pathlib import Path

nb_path = Path('deploy_kernel_midx/casmi26-v4n-engine-fusion-union-lb-0-399.ipynb')
nb = json.loads(nb_path.read_text())

nb['cells'][0]['source'] = [
    "# CASMI26 LB 0.480+ | Dual-Neural Z-Score Ensemble (Saner v1 + Scorer v2) + MIDX Exact Dedup Fill\n",
    "\n",
    "**Version 24**: Dual-Neural Z-Score Standardization Ensemble (Model A: `sanpier/casmi-spectrum-model-v1`, Model B: `bgunshui/casmi26-scorer-v2-model`).\n",
    "Ranks 2–25 are re-ranked using per-molecule z-score standardization `0.5*z(model_a) + 0.5*z(model_b)` combined with PubChem documentation prior and metric-exact InChIKey14 deduplication.\n",
    "Rank 1 is strictly guarded by the physical fragment re-scorer on V2.\n"
]

nb_path.write_text(json.dumps(nb, indent=1))
print("Prepared Cell 0 cleanly.")

