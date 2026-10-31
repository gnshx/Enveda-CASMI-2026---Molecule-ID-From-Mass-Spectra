#!/bin/bash
# Auto-monitor, submit v18, then push v4g CUSTOM kernel

KERNEL_SLUG="nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399"
COMPETITION="enveda-CASMI26-molecule-id-mass-spectra"
REPO_DIR="/home/dlcv/Enveda-CASMI-2026"

echo "[$(date)] Monitoring kernel v18..."

while true; do
    STATUS=$(kaggle kernels status "$KERNEL_SLUG" 2>&1)
    echo "[$(date)] Status: $STATUS"
    
    if echo "$STATUS" | grep -q "COMPLETE"; then
        echo "[$(date)] ✅ KERNEL COMPLETE! Downloading output..."
        
        mkdir -p /tmp/kernel_v18_out
        cd /tmp/kernel_v18_out
        kaggle kernels output "$KERNEL_SLUG" 2>&1
        
        if [ -f "submission.csv" ]; then
            echo "[$(date)] Submitting submission.csv..."
            RESULT=$(kaggle competitions submit -c "$COMPETITION" \
              -f submission.csv \
              -m "Plan A v18: Library Gate Fix + PC_JOIN_N=50 + POP_MU=0.25 + Surrogates" 2>&1)
            echo "[$(date)] Submit result: $RESULT"
            
            # Update scores
            sleep 5
            kaggle competitions submissions -c "$COMPETITION" | head -5
            
            echo "[$(date)] 🚀 Now pushing v4g CUSTOM (Rozen fork, w_mid=0.75)..."
            cd "$REPO_DIR"
            kaggle kernels push -p deploy_kernel_v4g 2>&1
            echo "[$(date)] v4g CUSTOM pushed!"
        else
            echo "[$(date)] ❌ No submission.csv found in output!"
            ls -la /tmp/kernel_v18_out/
        fi
        
        echo "[$(date)] Done. Exiting."
        break
        
    elif echo "$STATUS" | grep -q "ERROR\|FAIL"; then
        echo "[$(date)] ❌ KERNEL FAILED: $STATUS"
        break
    fi
    
    echo "[$(date)] Still running, checking again in 90s..."
    sleep 90
done
