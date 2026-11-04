#!/usr/bin/env python3
import json
from pathlib import Path

v25_nb_path = Path('deploy_kernel_v25/casmi26-v4n-engine-fusion-union-lb-0-399.ipynb')
nb = json.loads(v25_nb_path.read_text())

nb['cells'][0]['source'] = [
    "# CASMI26 LB 0.485+ | Tri-Neural Multi-Capacity Backbone + LOTUS 150k Knowledge Base + MIDX Exact Dedup Fill\n",
    "\n",
    "**Version 25**: Tri-Neural Multi-Capacity SOTA Ensemble + LOTUS Knowledge Base:\n",
    "- Tri-Neural Backbone: Model 1 (`sanpier v1`, 0.50) + Model 2 (`bgunshui v2`, 0.25) + Model 3 (`bgunshui v5` wider 320-dim, 0.25)\n",
    "- LOTUS 150k Knowledge Base (`qwertykeypadapple/casmi26-lotus-candidates`): 150k natural products with monoisotopic mass matching (15 ppm)\n",
    "- Calibrated Documentation Prior: `POP_W=0.10, SPEC_W=0.08, NP_W=0.06, LIB_HIT_GATE=0.82`\n",
    "- In-Pool InChIKey14 Skeletal Deduplication + Outsider Filling + LOTUS Natural Product Vacant Slot Backfill\n",
    "Rank 1 is strictly guarded by the physical fragment re-scorer on V2.\n"
]

v25_nb_path.write_text(json.dumps(nb, indent=1))
print("Version 25 Cell 0 cleaned.")
