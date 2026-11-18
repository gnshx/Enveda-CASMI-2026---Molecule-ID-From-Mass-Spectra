#!/usr/bin/env python3
import os, sys, time, subprocess

PROXY = "http://176.111.37.5:39811"
KAGGLE = "/home/dlcv/anaconda3/bin/kaggle"
SLUG = "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMP = "enveda-CASMI26-molecule-id-mass-spectra"

env = os.environ.copy()
env["https_proxy"] = PROXY
env["http_proxy"] = PROXY
env["PATH"] = f"/home/dlcv/anaconda3/bin:{env.get('PATH', '')}"

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

log("Starting background watcher for v41...")
while True:
    try:
        res = subprocess.run([KAGGLE, "kernels", "status", SLUG], capture_output=True, text=True, env=env, timeout=60)
        out = (res.stdout + res.stderr).strip()
        log(f"Kernel status: {out[:120]}")
        if "complete" in out.lower() and "error" not in out.lower():
            log("Kernel 41 COMPLETE! Submitting to competition...")
            sub_res = subprocess.run([
                KAGGLE, "competitions", "submit",
                "-c", COMP,
                "-f", "submission.csv",
                "-k", SLUG,
                "-v", "41",
                "-m", "Version 41: Maximum Regime B Promotion (0.90 w_top) + Maximum Pop Prior (0.40) + SPEC_W (0.15)"
            ], capture_output=True, text=True, env=env, timeout=120)
            log(f"Submit output: {sub_res.stdout} {sub_res.stderr}")
            break
        elif "error" in out.lower() and "complete" not in out.lower():
            log(f"Kernel error encountered: {out}")
            break
    except Exception as e:
        log(f"Error checking status: {e}")
    time.sleep(60)

log("Watcher finished.")
