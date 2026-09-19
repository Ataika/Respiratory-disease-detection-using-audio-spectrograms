#!/bin/zsh
set -e
cd "/Users/ataika/Library/Mobile Documents/com~apple~CloudDocs/Desktop/Thesis Diploma/Respiratory-disease-detection-using-audio-spectrograms"
export SSL_CERT_FILE="$(./venv/bin/python -c 'import certifi; print(certifi.where())')"
export MPLCONFIGDIR=results/.mplconfig

echo "=== resuming: seed 42 already complete, starting from seed 0 ==="

for SEED in 0 1 2 3; do
  echo "=== seed $SEED ==="
  ./venv/bin/python scripts/train_icbhi_baseline.py \
    --model-name audiomae --epochs 8 --batch-size 4 --learning-rate 5e-5 \
    --disable-weighted-sampler --seed "$SEED" \
    --checkpoint-path "results/checkpoints/final_audiomae_no_sampler_seed${SEED}_best.pth" \
    > "results/.mplconfig/log_seed_audiomae_${SEED}.txt" 2>&1
  echo "=== seed $SEED done ==="
done

echo "=== exporting predictions + recomputing full statistics for all 3 models ==="
./venv/bin/python scripts/export_val_predictions.py --model-name audiomae \
  --checkpoint results/checkpoints/final_audiomae_no_sampler_best.pth \
  --out results/predictions/audiomae_no_sampler_val.npz \
  >> results/.mplconfig/log_audiomae_overnight_main.txt 2>&1

./venv/bin/python scripts/compute_advanced_metrics.py \
  >> results/.mplconfig/log_audiomae_overnight_main.txt 2>&1

./venv/bin/python scripts/aggregate_multiseed_results.py \
  >> results/.mplconfig/log_audiomae_overnight_main.txt 2>&1

echo "ALL DONE $(date)" >> results/.mplconfig/log_audiomae_overnight_main.txt
