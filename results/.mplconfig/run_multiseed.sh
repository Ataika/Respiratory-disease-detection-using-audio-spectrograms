#!/bin/zsh
# Multi-seed robustness for the two main CNN baselines.
# Writes all rows to results/seed_robustness.csv (separate from final registry).
set -e
cd "/Users/ataika/Desktop/Thesis Diploma/Respiratory-disease-detection-using-audio-spectrograms"
export MPLCONFIGDIR=results/.mplconfig
PY=venv/bin/python
CSV=results/seed_robustness.csv
TMP=results/checkpoints/tmp_seed.pth

for MODEL in resnet50 efficientnet_b3; do
  for SEED in 42 0 1 2 3; do
    echo "=== $MODEL seed=$SEED  $(date) ==="
    $PY scripts/train_icbhi_baseline.py --model-name $MODEL --epochs 10 --batch-size 8 \
      --learning-rate 1e-4 --seed $SEED \
      --checkpoint-path $TMP --results-csv $CSV \
      > results/.mplconfig/log_seed_${MODEL}_${SEED}.txt 2>&1
  done
done
echo "MULTISEED_DONE $(date)"
