#!/usr/bin/env python3
"""
scripts/build_v42.py
=============================================================================
CASMI 2026 — V42 Builder: Restore Exact V24 0.480 PB Control + Stage Execution Audit

Builds `deploy_kernel_v42/` starting directly from `deploy_kernel_midx/` (V24 PB):
  1. Preserves exact V24 scoring configuration:
     CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)
     POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02
  2. Fixes Cell 3 misleading comment so documentation matches actual code.
  3. Adds deterministic stage execution & hash auditing (`v42_execution_audit.json`):
     - Records PUB_BUNDLE / OURS_BUNDLE selection and checkpoint SHA-256
     - Logs candidate admissions, rank-1 moves, deduplication swaps, outsider fills
     - Records SHA-256 hashes of every intermediate and final submission artifact
     - Fails loudly on unexpected silent fallback or marks fallback reasons explicitly
=============================================================================
"""

import json
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT / "deploy_kernel_midx"
OUT_DIR = ROOT / "deploy_kernel_v42"


def build_v42() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nb = json.loads((BASE_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb").read_text(encoding="utf-8"))
    meta = json.loads((BASE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    # Cell 0: Accurate header
    nb["cells"][0]["source"] = [
        "# CASMI26 V42 | Exact V24 0.480 PB Control + Stage Execution Audit\n",
        "\n",
        "**Version 42**: Exact restoration of the verified V24 Personal Best (`0.4800` LB) with full per-stage execution, model-load, rank-1 movement, and SHA-256 output auditing.\n",
    ]

    # Cell 3: Align comment with exact V24 configuration (w_top=0.25, w_mid=0.75, rescue=0)
    c3 = "".join(nb["cells"][3]["source"])
    old_comment = "#   'CUSTOM' uses the Pure Rank 1 Shield (w_top=0.00, w_mid=0.75, rescue=0)"
    new_comment = "#   'CUSTOM' exact V24 0.480 PB control (w_top=0.25, w_mid=0.75, rescue=0)"
    if old_comment in c3:
        c3 = c3.replace(old_comment, new_comment, 1)
    assert "CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)" in c3, "Expected exact V24 CUSTOM config"
    assert "POP_W, SPEC_W, NP_W = 0.10, 0.05, 0.02" in c3, "Expected exact V24 prior weights"
    nb["cells"][3]["source"] = [l + "\n" for l in c3.splitlines()]

    # Cell 13: Record neural bundle paths, swap status, and checkpoint SHA-256
    c13 = "".join(nb["cells"][13]["source"])
    audit_c13 = (
        "\n# ---- V42 Stage Audit: Neural Bundles & Control Hash ----\n"
        "def _sha256_file(p):\n"
        "    h = hashlib.sha256()\n"
        "    with open(p, 'rb') as f:\n"
        "        for chunk in iter(lambda: f.read(1 << 20), b''):\n"
        "            h.update(chunk)\n"
        "    return h.hexdigest()\n"
        "V42_AUDIT = {\n"
        "    'version': 'V42_V24_PB_CONTROL',\n"
        "    'custom_cfg': dict(CUSTOM),\n"
        "    'prior_weights': {'POP_W': POP_W, 'SPEC_W': SPEC_W, 'NP_W': NP_W, 'LIB_HIT_GATE': LIB_HIT_GATE},\n"
        "    'pub_bundle': str(PUB_BUNDLE),\n"
        "    'ours_bundle': str(OURS_BUNDLE),\n"
        "    'ours_swapped': str(OURS_BUNDLE) != str(PUB_BUNDLE),\n"
        "    'pub_model_sha256': _sha256_file(Path(PUB_BUNDLE) / 'final_model.pt') if (Path(PUB_BUNDLE) / 'final_model.pt').is_file() else None,\n"
        "    'ours_model_sha256': _sha256_file(Path(OURS_BUNDLE) / 'final_model.pt') if (Path(OURS_BUNDLE) / 'final_model.pt').is_file() else None,\n"
        "    'neural_ok': bool(NEURAL_OK),\n"
        "    'outsiders_ok': bool(OUTSIDERS_OK),\n"
        "    'control_v2_sha256': _sha256_file(ROOT / 'submission_control_V2.csv') if (ROOT / 'submission_control_V2.csv').is_file() else None,\n"
        "}\n"
        "if not V42_AUDIT['ours_swapped']:\n"
        "    print('AUDIT WARNING: _swap_to_ours did NOT swap away from PUB_BUNDLE!', flush=True)\n"
        "print('V42_STAGE13_AUDIT:', json.dumps(V42_AUDIT), flush=True)\n"
    )
    c13 = c13.rstrip() + "\n" + audit_c13
    nb["cells"][13]["source"] = [l + "\n" for l in c13.splitlines()]

    # Cell 17: Persist complete stage audit including fallback reasons and final output hash
    c17 = "".join(nb["cells"][17]["source"])
    audit_c17 = (
        "\n# ---- V42 Final Stage Audit ----\n"
        "try:\n"
        "    _audit = dict(globals().get('V42_AUDIT', {}))\n"
        "    _audit['parallax_diagnostics'] = globals().get('PARALLAX_DIAGNOSTICS', {})\n"
        "    _audit['provenance_report'] = globals().get('PROVENANCE_REPORT', {})\n"
        "    _audit['merge_report'] = {\n"
        "        k: MERGE_REPORT.get(k) for k in (\n"
        "            'status', 'rows', 'merged', 'leader_moves', 'ranks_reordered',\n"
        "            'dedup_swaps', 'outsider_fills', 'old_leader_pinned',\n"
        "            'lib_hit_molecules', 'output_equals_custom', 'output_sha256', 'exception'\n"
        "        ) if k in MERGE_REPORT\n"
        "    }\n"
        "    _audit['final_submission_sha256'] = hashlib.sha256((ROOT / 'submission.csv').read_bytes()).hexdigest()\n"
        "    (ROOT / 'v42_execution_audit.json').write_text(json.dumps(_audit, indent=2, default=str))\n"
        "    print('V42_FINAL_AUDIT:', json.dumps(_audit, default=str), flush=True)\n"
        "    if MERGE_REPORT.get('status') not in ('applied', 'skipped_fast_commit'):\n"
        "        raise RuntimeError(f'V42 Audit aborted silent fallback: {MERGE_REPORT}')\n"
        "except Exception as _audit_exc:\n"
        "    print('V42_AUDIT_ERROR:', repr(_audit_exc), flush=True)\n"
        "    raise\n"
    )
    c17 = c17.rstrip() + "\n" + audit_c17
    nb["cells"][17]["source"] = [l + "\n" for l in c17.splitlines()]

    # Validate syntax of every Python code cell
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            src = "".join(cell["source"])
            ast.parse(src)

    nb_out = OUT_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
    nb_out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Built {nb_out} ({nb_out.stat().st_size:,} bytes) — all code cells AST-verified.")
    return nb_out


if __name__ == "__main__":
    build_v42()
