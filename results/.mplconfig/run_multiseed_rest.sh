#!/bin/zsh
set -e
cd "/Users/ataika/Desktop/Thesis Diploma/Respiratory-disease-detection-using-audio-spectrograms"
export MPLCONFIGDIR=results/.mplconfig
PY=venv/bin/python
CSV=results/seed_robustness.csv
TMP=results/checkpoints/tmp_seed.pth
run() {
  echo "=== $1 seed=$2 $(date) ==="
  $PY scripts/train_icbhi_baseline.py --model-name $1 --epochs 10 --batch-size 8 \
    --learning-rate 1e-4 --seed $2 --checkpoint-path $TMP --results-csv $CSV \
    > results/.mplconfig/log_seed_$1_$2.txt 2>&1
}
run resnet50 2
run resnet50 3
run efficientnet_b3 2
run efficientnet_b3 3
echo "REST_MULTISEED_DONE $(date)"
