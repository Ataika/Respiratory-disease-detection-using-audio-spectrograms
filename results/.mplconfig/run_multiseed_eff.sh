#!/bin/zsh
set -e
cd "/Users/ataika/Desktop/Thesis Diploma/Respiratory-disease-detection-using-audio-spectrograms"
export MPLCONFIGDIR=results/.mplconfig
PY=venv/bin/python
CSV=results/seed_robustness.csv
TMP=results/checkpoints/tmp_seed.pth
for SEED in 42 0 1; do
  echo "=== efficientnet_b3 seed=$SEED $(date) ==="
  $PY scripts/train_icbhi_baseline.py --model-name efficientnet_b3 --epochs 10 --batch-size 8 \
    --learning-rate 1e-4 --seed $SEED --checkpoint-path $TMP --results-csv $CSV \
    > results/.mplconfig/log_seed_eff_${SEED}.txt 2>&1
done
echo "EFF_MULTISEED_DONE $(date)"
