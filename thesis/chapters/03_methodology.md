# Chapter 3 — Methodology

This chapter describes how five heterogeneous respiratory-sound datasets
were unified into a single labeling scheme and used to train three
architectures under a protocol designed to answer three questions: (1)
does a self-supervised transformer outperform supervised CNNs on ICBHI
once seed variance is accounted for; (2) how much of that performance
survives a shift in recording equipment and patient population; (3)
which acoustic regions drive correct versus incorrect predictions.

## 3.1 Datasets

Five publicly available respiratory-sound datasets were used, split
between two roles: three as training/validation data, and two as
completely unseen, zero-shot cross-domain evaluation sets.

| Dataset | Records | Role | Label granularity |
|---|---|---|---|
| ICBHI 2017 | 6,898 breathing cycles | Train / validation | normal, crackle, wheeze, both |
| HF Lung V1 | 52,444 segments | Parsed, not used in training (see 3.1.1) | normal, wheeze, crackle, other |
| KAUH | 336 recordings | Parsed, not used in training (see 3.1.1) | normal / abnormal |
| BRACETS | 3,291 recordings | Unseen cross-domain eval | binary: healthy / pathological |
| SPRSound | 1,772 recordings | Unseen cross-domain eval | normal, crackle, wheeze, both |

**ICBHI 2017** [cite ICBHI dataset paper] is the primary training source.
Each recording is annotated at the level of individual respiratory
cycles, with per-cycle start/end timestamps and binary crackle/wheeze
flags rather than a single file-level label. This project's parser
(`src/data/parsers/icbhi.py`) preserves that granularity: one record is
generated per annotated breathing cycle rather than one per file, which
is a finer unit than a simple file-level majority-vote label would give.

**BRACETS** and **SPRSound** are held out entirely from training. BRACETS
provides only a binary healthy/pathological label; SPRSound uses the
same four-class scheme as ICBHI, so its predictions require no label
remapping. Both are used exclusively for the zero-shot cross-domain
experiments in Chapter 4.

### 3.1.1 A scope limitation: single-dataset training

Although HF Lung V1 and KAUH were fully parsed, unit-tested, and
integrated into the unified label scheme (52,444 and 336 records
respectively), all final training runs reported in Chapter 4 use ICBHI
alone. This is a deliberate scope reduction under time constraints
rather than a technical limitation of the pipeline — the parsers and
data loaders support multi-dataset training without modification. This
is stated explicitly here, rather than left implicit, because it
directly affects how the cross-domain results in Chapter 4 should be
read: the model was exposed to exactly one recording environment during
training, which is itself a partial explanation for the cross-domain
collapse reported in Section 4.6.

Similarly, the SPRSound evaluation set uses only its `train` split
(1,772 recordings); the dataset's `valid` split was not available in
the local mirror used for this project and was excluded rather than
substituted with a smaller subset.

## 3.2 Unified label scheme

All training-side labels are mapped onto a common four-class scheme:

| Label | Class | Meaning |
|---|---|---|
| 0 | normal | No adventitious sounds |
| 1 | crackle | Discontinuous, explosive sounds |
| 2 | wheeze | Continuous, high-pitched sounds |
| 3 | both | Crackle and wheeze co-occurring |

For ICBHI, this scheme maps directly onto the dataset's native
crackle/wheeze binary flags per cycle: `(0,0)→normal`, `(1,0)→crackle`,
`(0,1)→wheeze`, `(1,1)→both`. BRACETS, which only distinguishes
healthy/pathological, is treated as a separate binary task at
evaluation time (Section 4.6) by collapsing model predictions
`{crackle, wheeze, both} → pathological`.

## 3.3 Preprocessing pipeline

Two preprocessing branches were implemented, since the CNN and
transformer architectures require different input representations from
the same underlying audio.

**Loading and normalization.** Each audio window is loaded at a
branch-specific sample rate (22,050 Hz for the CNN branch, 16,000 Hz
for the AudioMAE branch), then normalized per-recording to zero mean
and unit variance (`instance_normalize`). This step exists specifically
because the five source datasets were recorded on different equipment
and in different clinical settings, which otherwise introduces
amplitude and gain differences unrelated to the acoustic content being
classified.

**Windowing.** Fixed-length windows are extracted starting at each
annotated cycle's onset — 5 seconds for the CNN branch, 10 seconds for
AudioMAE — and zero-padded if the recording is shorter than the window.

**CNN branch.** A mel spectrogram is computed (`n_mels=128,
hop_length=512, n_fft=2048`), converted to log scale, min-max normalized
to [0, 1], replicated across three channels, and normalized with
ImageNet channel statistics — since both CNN backbones (ResNet50,
EfficientNet-B3) are ImageNet-pretrained and expect that normalization.

**AudioMAE branch.** A log-mel filterbank is computed via
`torchaudio.compliance.kaldi.fbank` (128 mel bins, matching AudioMAE's
pretraining configuration), zero-padded or truncated to 1,024 frames,
and normalized with the AudioSet mean/std statistics used during
AudioMAE's pretraining.

## 3.4 Data augmentation

SpecAugment [Park et al., 2019] is applied to the CNN branch: one time
mask (up to 80 frames) and one frequency mask (up to 30 mel bins) per
spectrogram, applied only during training. Its effect on this dataset
is examined directly in Section 4.3 rather than assumed — the ablation
there shows it raises accuracy but *reduces* macro F1, by pushing
predictions further toward the majority `normal` class.

## 3.5 Class imbalance handling

ICBHI is imbalanced toward `normal` (roughly half the cycles). Two
independent mechanisms were implemented and evaluated separately:

- **Weighted random sampling**: training batches are drawn with
  per-class inverse-frequency weights, so each class is seen roughly
  equally often regardless of its true prevalence.
- **Class-weighted loss**: `CrossEntropyLoss` is weighted by
  inverse class frequency instead of (or in addition to) resampling.

These were tested as separate configurations, not combined by default:
an early experiment combining both simultaneously degraded validation
performance relative to either alone, likely because the two mechanisms
compound rather than complement each other's correction for the same
imbalance. The results in Section 4.3 report weighted sampling alone
(the default for the "tuned" configuration) against class-weighted loss
alone (sampler disabled) as two separate, comparable configurations.

## 3.6 Model architectures

Three architectures were trained and compared, spanning supervised CNNs
and a self-supervised transformer:

**ResNet50** and **EfficientNet-B3**, both ImageNet-pretrained
(`torchvision`), with their final classification layer replaced by a
`Dropout(p=0.3) → Linear(num_classes=4)` head. Using pretrained ImageNet
weights on mel-spectrogram "images" is an established transfer-learning
approach in audio classification: the low-level filters learned on
natural images (edge and texture detectors) transfer usefully to the
time-frequency textures in a spectrogram, even though the domain is
acoustic rather than visual.

**AudioMAE**, a Vision-Transformer-Base backbone pretrained via masked
autoencoding on AudioSet (accessed via `timm`,
`hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m`), fine-tuned
end-to-end with a linear classification head.

### 3.6.1 Substituting AST with AudioMAE

The original architecture plan specified the Audio Spectrogram
Transformer (AST) [Gong et al., 2021] as the transformer branch,
following its established track record on ICBHI [Bae et al., 2023].
AudioMAE was substituted because its pretraining objective (masked
autoencoding, reconstructing masked spectrogram patches) does not
depend on AudioSet's label taxonomy aligning with respiratory-sound
acoustic characteristics, unlike AST's supervised pretraining on
sound-event tags. This keeps the comparison structurally analogous to
prior CNN-vs-transformer ICBHI studies while testing a self-supervised
pretraining signal that is comparatively less explored on this dataset
than supervised AST fine-tuning.

> **Author's note on this justification.** The paragraph above is a
> defensible technical rationale for the substitution, constructed
> during later analysis — it is not a transcript of the reasoning at the
> time the decision was made, which was not recorded in the project's
> development journal. If asked in the defense *why* this substitution
> happened, answer honestly (e.g., pretrained-weight availability,
> practical convenience, or genuine interest in self-supervised
> pretraining — whichever is actually true) rather than presenting this
> paragraph as historical fact. Replace this note once the real
> reasoning is confirmed.

## 3.7 Training procedure

All models are trained with Adam (`lr=1e-4` for the CNNs, `lr=5e-5` for
AudioMAE — a lower rate given the larger pretrained backbone),
`weight_decay=1e-4`, and `ReduceLROnPlateau` scheduling
(`factor=0.5, patience=1` epoch on validation loss). Early stopping
halts training after 2 consecutive epochs without validation-loss
improvement (`patience=2`), and the checkpoint corresponding to the
*best* validation loss — not the final epoch — is retained for
evaluation; training accuracy continues to rise well past this point
(Section 4.2), so retaining the final-epoch checkpoint would silently
evaluate an overfit model.

CNN models are trained for up to 10 epochs with `batch_size=8`; AudioMAE
for up to 8 epochs with `batch_size=4` (a smaller batch given its larger
memory footprint) and weighted sampling disabled. Unlike both CNN
configurations, **AudioMAE training uses neither weighted sampling nor
class-weighted loss** — no class-imbalance handling was applied to this
branch. This was not a deliberate ablation choice and is noted here as
a methodological gap: it means AudioMAE's results (Section 4.4) are not
directly comparable to the CNN "tuned" configuration on this dimension,
and its performance may be partly confounded by imbalance rather than
purely reflecting architecture. If time allows, re-running AudioMAE
with weighted sampling enabled would remove this confound.

## 3.8 Evaluation protocol

**Split.** All train/validation splits are patient-level, not
recording-level: every breathing cycle from a given patient is assigned
entirely to either the training or validation set (never both), with a
15% validation ratio (`create_splits`, seed 42). This prevents a common
data-leakage failure mode in this domain — cycles from the same patient
recording share acoustic characteristics (voice, chest cavity resonance,
recording position) that a model could exploit to "recognize the
patient" rather than the pathology, inflating validation performance
without corresponding real-world generalization.

**Metrics.** Accuracy alone is reported to be insufficient throughout
this thesis, since ICBHI's class imbalance means a model that always
predicts `normal` scores roughly 50% accuracy. The full evaluation
protocol therefore reports: macro F1 (equal weight per class regardless
of support), the per-class confusion matrix, the official ICBHI
challenge score — Sensitivity/Specificity on the binary
normal-vs-abnormal collapse, Score = (Se+Sp)/2 — in-domain macro
AUC-ROC (one-vs-rest), and Expected Calibration Error (ECE) with a
reliability diagram, quantifying whether the model's confidence tracks
its actual accuracy. Statistical comparison between models uses
McNemar's test (paired, same validation samples) on prediction
correctness, and DeLong's test on the binary abnormal-probability AUC,
alongside a 5-seed robustness check and bootstrap 95% confidence
intervals on macro F1 — full results in Chapter 4.
