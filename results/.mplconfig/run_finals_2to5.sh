#!/bin/zsh
# Runs final experiments 2..5 sequentially. Logs to results/.mplconfig/.
set -e
cd "/Users/ataika/Desktop/Thesis Diploma/Respiratory-disease-detection-using-audio-spectrograms"
export MPLCONFIGDIR=results/.mplconfig
PY=venv/bin/python
L=results/.mplconfig

echo "[2/5] efficientnet_b3_tuned"
$PY scripts/train_icbhi_baseline.py --model-name efficientnet_b3 --epochs 10 --batch-size 8 --learning-rate 1e-4 \
  --checkpoint-path results/checkpoints/final_efficientnet_b3_tuned_best.pth > $L/log_final_efficientnet_b3_tuned.txt 2>&1

echo "[3/5] audiomae_no_sampler"
$PY scripts/train_icbhi_baseline.py --model-name audiomae --epochs 8 --batch-size 4 --learning-rate 5e-5 --disable-weighted-sampler \
  --checkpoint-path results/checkpoints/final_audiomae_no_sampler_best.pth > $L/log_final_audiomae_no_sampler.txt 2>&1

echo "[4/5] resnet50_specaugment"
$PY scripts/train_icbhi_baseline.py --model-name resnet50 --epochs 10 --batch-size 8 --learning-rate 1e-4 --use-spec-augment \
  --checkpoint-path results/checkpoints/final_resnet50_specaugment_best.pth > $L/log_final_resnet50_specaugment.txt 2>&1

echo "[5/5] resnet50_class_weighted_no_sampler"
$PY scripts/train_icbhi_baseline.py --model-name resnet50 --epochs 10 --batch-size 8 --learning-rate 1e-4 --use-class-weights --disable-weighted-sampler \
  --checkpoint-path results/checkpoints/final_resnet50_class_weighted_no_sampler_best.pth > $L/log_final_resnet50_class_weighted_no_sampler.txt 2>&1

echo "ALL_DONE_2to5"
