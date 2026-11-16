#!/usr/bin/env python3
"""
Deploy and Submit Version 31: Apex Sovereign Frontier
- In-Test Molecular Network Reranker (N-1c, LAM=0.35, TAU=0.60)
- Optimized External Spectral Reference Library (TAU=0.80, K=4, LIB_TAU=0.90)
- Learned Logistic Re-Ranker on Enveda-180 panel (RERANK_GATE=1.01)
- Multi-Neural Ensemble + Bruker timsTOF Mass Prior + InChIKey14 Skeletal Dedup
Target Public Leaderboard: 0.488 - 0.492+
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

os.environ["https_proxy"] = "http://176.111.37.5:39811"
os.environ["http_proxy"] = "http://176.111.37.5:39811"

ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = ROOT / "deploy_kernel_v31"
KERNEL_SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION = "enveda-CASMI26-molecule-id-mass-spectra"
VERSION_LABEL = "Version 31: Apex Sovereign Frontier (N-1c In-Test Molecular Network + TAU=0.80 ExtLib + Learned Logistic Reranker) (Target LB 0.488+)"
LOG_FILE = ROOT / "logs" / "runner_v31.log"

def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_cmd(cmd, check=True):
    env = os.environ.copy()
    env["https_proxy"] = "http://176.111.37.5:39811"
    env["http_proxy"] = "http://176.111.37.5:39811"
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if check and res.returncode != 0:
        log(f"Command failed ({res.returncode}): {cmd}\nStderr: {res.stderr.strip()}")
        raise RuntimeError(f"Command failed: {cmd}\n{res.stderr.strip()}")
    return res

def get_current_kernel_version():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    req = ApiGetKernelRequest()
    req.user_name = "nukaladevisaiganesh"
    req.kernel_slug = "casmi26-v4n-engine-fusion-union-lb-0-399"
    res = client.kernels.kernels_api_client.get_kernel(req)
    current_ver = getattr(res.metadata, "current_version_number", getattr(res.metadata, "currentVersionNumber", 30))
    log(f"Kaggle reports current kernel version number: {current_ver}")
    return int(current_ver)

def main():
    log("=== Starting Version 31 Deployment Pipeline ===")
    
    # 1. Determine next version number
    current_ver = get_current_kernel_version()
    target_ver = current_ver + 1
    log(f"Deploying as Version {target_ver}...")
    
    # 2. Push kernel to Kaggle
    log(f"Pushing notebook from {DEPLOY_DIR} to {KERNEL_SLUG}...")
    res = run_cmd(f"kaggle kernels push -p {DEPLOY_DIR}")
    log(f"Push result: {res.stdout.strip()}")
    
    # 3. Wait for fast commit run to complete
    log("Waiting for Kaggle commit run to complete...")
    start_time = time.time()
    while time.time() - start_time < 3600: # 1 hour max wait
        time.sleep(15)
        st_res = run_cmd(f"kaggle kernels status {KERNEL_SLUG}", check=False)
        st_out = st_res.stdout.strip()
        elapsed = (time.time() - start_time) / 60.0
        
        if "KernelWorkerStatus.COMPLETE" in st_out or "COMPLETE" in st_out:
            log(f"Kernel execution finished COMPLETE in {elapsed:.1f} minutes!")
            break
        elif "KernelWorkerStatus.FAILED" in st_out or "FAILED" in st_out or "ERROR" in st_out:
            log(f"Kernel run FAILED: {st_out}")
            raise RuntimeError(f"Kernel run failed: {st_out}")
        elif "QUEUED" in st_out:
            log(f"Status: QUEUED (elapsed {elapsed:.1f}m)")
        elif "RUNNING" in st_out:
            log(f"Status: RUNNING (elapsed {elapsed:.1f}m)")
        else:
            log(f"Status: {st_out} (elapsed {elapsed:.1f}m)")
            
    # 4. Verify version number after push
    new_ver = get_current_kernel_version()
    log(f"Verified new kernel version: {new_ver}")
    
    # 5. Submit to competition
    log(f"Submitting Version {new_ver} to competition '{COMPETITION}'...")
    sub_cmd = (
        f"kaggle competitions submit -c {COMPETITION} "
        f"-f submission.csv "
        f"-k {KERNEL_SLUG} "
        f"-v {new_ver} "
        f'-m "{VERSION_LABEL}"'
    )
    sub_res = run_cmd(sub_cmd)
    sub_output = sub_res.stdout.strip()
    log(f"Competition submit output: {sub_output}")
    
    # Parse submission ref
    sub_ref = None
    for token in sub_output.split():
        if token.isdigit() and len(token) >= 7:
            sub_ref = token
            break
    log(f"Submission Ref: {sub_ref}")
    
    # 6. Append to DAILY_JOURNEY.md
    journey_path = ROOT / "progress" / "DAILY_JOURNEY.md"
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    entry = rf"""
### Run Record: Version {new_ver} (Apex Sovereign Frontier)
- **Timestamp**: {ts}
- **Submission Ref**: `{sub_ref or 'Pending'}`
- **Configuration**: {VERSION_LABEL}
- **Innovations Deployed**:
  * In-Test Molecular Network Reranker (`N-1c`): Modified cosine $\ge 0.60$, $k \ge 6$ peak matching, Tanimoto propagation ($\lambda = 0.35$).
  * Tuned External Reference Library (`Apex ExtLib`): 324,884 spectra at `TAU = 0.80`, `K_PEAKS = 4`, `LIB_TAU = 0.90`.
  * Learned Logistic Re-Ranker: Trained on Enveda-180 panel (`RERANK_GATE = 1.01`).
  * Bruker timsTOF mass accuracy prior (`V8 = 'MASS'`) + InChIKey14 deduplication.
  * Fast single-pass outsider scoring preventing the 9-hour container timeout trap.
- **Status**: Submitted (scoring in progress)
- **Target Public LB Score**: **`0.488 - 0.492+`**
- **Artifacts**: Deployed from `deploy_kernel_v31/`
"""
    with open(journey_path, "a") as f:
        f.write(entry)
    log("Appended run record to progress/DAILY_JOURNEY.md")
    
    # 7. Update ROADMAP_AND_PROGRESS.md
    roadmap_path = ROOT / "progress" / "ROADMAP_AND_PROGRESS.md"
    roadmap_entry = f"\n- **Version {new_ver}** (Ref `{sub_ref}`): Apex Sovereign Frontier (N-1c Molecular Network + ExtLib TAU=0.80 + Learned Logistic Reranker) -> **Target LB 0.488 - 0.492+** [DEPLOYED & SCORING]\n"
    with open(roadmap_path, "a") as f:
        f.write(roadmap_entry)
    log("Appended run record to progress/ROADMAP_AND_PROGRESS.md")
    
    # 8. Sync Git
    git_msg = f"feat(phase-18): deploy Version {new_ver} Apex Sovereign Frontier (N-1c + ExtLib 0.80 + Logistic Reranker) (Ref {sub_ref})"
    log(f"Syncing git: {git_msg}")
    run_cmd("git add deploy_kernel_v31/ scripts/deploy_and_submit_v31.py progress/DAILY_JOURNEY.md progress/ROADMAP_AND_PROGRESS.md")
    run_cmd(f'git commit -m "{git_msg}"', check=False)
    git_res = run_cmd("git push origin main", check=False)
    log(f"Git push result: {git_res.stdout.strip()}")
    
    log(f"=== Version {new_ver} Deployed Successfully! ===")

if __name__ == "__main__":
    main()
