# Chapter 4 — Experiments & Results

## 4.1 Setup

All final experiments run on an Apple Silicon Mac (MPS backend); early
exploratory runs were on an 8 GB MacBook Air M1, which constrained batch
size and motivated the Windows/CUDA fallback path documented in
`docs/WINDOWS_SETUP.md` (not ultimately needed once migrated to a more
capable machine). All reported numbers use the patient-level ICBHI
validation split (seed 42, Section 3.8), 1,119 breathing cycles.

## 4.2 Baseline comparison: ResNet50 vs. EfficientNet-B3

| Model | Best epoch | Val loss | Val acc | Val macro F1 |
|---|---|---|---|---|
| ResNet50 | 1 | 1.176 | 0.586 | 0.438 |
| EfficientNet-B3 | 2 | 1.125 | 0.542 | 0.434 |

Both models learn useful acoustic features from ICBHI spectrograms, and
both overfit early: training accuracy exceeds 85% by epoch 5 in both
cases while validation loss stops improving after epoch 1-2. This
motivated the regularization pass in Section 4.3 before any further
architecture comparison was treated as meaningful.

## 4.3 Regularization and tuning

Three tuned configurations of ResNet50 (10 epochs, early stopping +
`ReduceLROnPlateau` in all three) are compared against each other,
isolating one change at a time relative to the `tuned` baseline
configuration (weighted sampler on, no class-weighted loss, no
SpecAugment):

| Configuration | Change vs. tuned | Best epoch | Val loss | Val acc | Val macro F1 |
|---|---|---|---|---|---|
| tuned (reference) | — | 3 | 1.225 | 0.580 | 0.458 |
| class-weighted, no sampler | sampler → loss | 2 | 1.158 | 0.508 | 0.426 |
| SpecAugment | + augmentation | 1 | 1.013 | 0.554 | 0.423 |

Two findings from this table:

**Weighted sampling outperforms class-weighted loss for this dataset.**
An earlier hypothesis (recorded during development) was that combining
weighted sampling with class-weighted loss over-corrected for imbalance
and that isolating class-weighted loss alone would reveal its "true"
benefit. The isolated result does not support that hypothesis: even
alone, class-weighted loss underperforms weighted sampling alone on both
val loss and macro F1. The practical implication for this dataset is
that balancing at the sampling level is the more effective mechanism.

**SpecAugment improves val loss and accuracy but *reduces* macro F1.**
This is not a contradiction — it is the accuracy-vs-macro-F1 tension the
evaluation protocol (Section 3.8) was designed to catch. SpecAugment
pushes predictions toward the majority `normal` class rather than
improving discrimination across all four classes equally.

The `both` class remains the weakest across all three configurations
without exception — a pattern that recurs in every subsequent
experiment in this chapter (Sections 4.4, 4.8) and is discussed as a
structural limitation in Section 4.11.

## 4.4 Three-way model comparison

| Model | Se | Sp | ICBHI Score | Macro F1 | AUC-ROC (macro OvR) | ECE |
|---|---|---|---|---|---|---|
| ResNet50 (tuned) | 0.408 | 0.752 | 0.580 | 0.458 | 0.743 | 0.190 |
| EfficientNet-B3 (tuned) | 0.506 | 0.568 | 0.537 | 0.464 | 0.777 | 0.114 |
| AudioMAE (no-sampler) | — | — | — | — | — | — |

**AudioMAE row intentionally left blank here** — see Section 4.4.1.

The ResNet50/EfficientNet-B3 comparison itself already illustrates why
a single headline metric is misleading: ResNet50 wins on ICBHI Score
(higher specificity), EfficientNet-B3 wins on macro F1 and AUC-ROC
(higher sensitivity, better overall discrimination), and pairwise
McNemar's test on the same validation samples shows their error
patterns *are* significantly different (p = 0.019) even though neither
model is clearly "better" by every metric. DeLong's test on the binary
abnormal-vs-normal AUC is borderline (p = 0.054) — consistent with, not
contradicting, the multi-seed finding in Section 4.5 that a single-run
comparison between these two architectures is close to the edge of
statistical distinguishability.

### 4.4.1 A note on AudioMAE's status in this draft

The historical single training run for AudioMAE (June 2026, prior
hardware) reported macro F1 = 0.484 — the best of the three
architectures, and the headline "self-supervised transformer wins"
result this thesis originally planned to report. After migrating to
new hardware, two independent verified single-process retraining
attempts (one interrupted mid-training, one allowed to run longer but
still not confirmed to reach the historical best epoch) both produced
substantially lower macro F1 (0.235–0.243). Given Section 4.5's own
finding — that single-run comparisons between architectures are not
trustworthy without a multi-seed check — it would be inconsistent to
report either the old high number or the new low numbers as "the"
AudioMAE result without first (a) confirming a fully-converged training
run on the current hardware, and (b) running it across multiple seeds,
exactly as was done for ResNet50/EfficientNet-B3. **This is left as an
open item, not silently resolved in either direction** — see Section
5.3 (Limitations) if it is not closed before submission.

## 4.5 Robustness: multi-seed analysis

Five independent seeds (42, 0, 1, 2, 3) for both ResNet50 and
EfficientNet-B3, all other hyperparameters fixed:

| Model | n | Val macro F1 (mean ± SD) | Min | Max |
|---|---|---|---|---|
| ResNet50 | 5 | 0.403 ± 0.040 | 0.361 | 0.460 |
| EfficientNet-B3 | 5 | 0.402 ± 0.028 | 0.354 | 0.437 |

The seed-to-seed spread within a single architecture (0.028–0.040) is
comparable to, and in ResNet50's case larger than, the entire gap
between the two architectures' means (Δ = 0.001). Framed against the
literature: Polanco-Martagón et al. [R2] explicitly flag the absence of
a variance analysis as a limitation of their own five-CNN comparison on
this same task family, citing computational cost. This result
demonstrates concretely what that absence risks — a single favorable or
unfavorable seed could flip which architecture appears to "win" a
head-to-head comparison, independent of any real architectural
advantage.

*(Note on seeding validity: `--seed` previously only controlled the
patient-level train/val split, not `torch.manual_seed()` — meaning even
the same seed value wasn't fully reproducible in the original runs.
This was fixed (Section 3.7); the five seeds above are the original,
pre-fix runs, so they reflect genuine independent randomness across
model init, sampler draws, and SpecAugment masks in addition to split
composition — if anything a *more* honest robustness check than a
"cleaner" experiment would have been, since it captures more of the
real sources of run-to-run variance a practitioner would actually see.)*

## 4.6 Cross-domain: BRACETS

The ICBHI-trained ResNet50 (tuned checkpoint) was evaluated zero-shot —
no fine-tuning, no domain adaptation — on BRACETS (3,291 recordings,
binary healthy/pathological), with model predictions collapsed to
binary via `{crackle, wheeze, both} → pathological`:

| | Value |
|---|---|
| Accuracy | 0.380 |
| Macro F1 | 0.366 |
| AUROC | 0.509 |

An AUROC of 0.509 is statistically indistinguishable from chance
(0.5): the model's confidence ranking of BRACETS recordings carries
essentially no information about which are actually pathological. This
is consistent with Kim et al. [R1]'s finding that respiratory-sound
classifiers absorb recording-equipment- and protocol-specific acoustic
signatures rather than clinically generalizable ones — quantified here
directly across two fully independent datasets rather than across
stethoscope types within one dataset, as in their study. Given the
scope limitation noted in Section 3.1.1 (training used ICBHI alone,
despite two additional in-domain datasets being available in the
pipeline), part of this collapse may also reflect narrow training
exposure rather than an inherent property of the architecture — a
question Section 5.4 (Future Work) returns to.

## 4.7 Cross-domain: SPRSound

The same checkpoint, evaluated zero-shot on SPRSound (1,772 recordings,
native 4-class labels, no remapping needed):

| | Value |
|---|---|
| Accuracy | 0.231 |
| Macro F1 | 0.187 |

Worse than BRACETS on every metric. The confusion matrix shows a
specific, interpretable failure mode: 1,064 of 1,303 truly-`normal`
SPRSound recordings are predicted `crackle` (recall on `normal` = 0.12;
recall on `crackle` = 0.92 — the model over-predicts `crackle`
almost indiscriminately on this dataset). We attribute the larger
collapse relative to BRACETS to a compounding factor beyond equipment:
**SPRSound is a pediatric dataset** (SJTU, China), while ICBHI is
predominantly adult, adding a population-level acoustic shift (airway
geometry, respiratory rate, recording context) on top of whatever
equipment/protocol shift both external datasets share with each other.

## 4.8 Interpretability: Grad-CAM

Grad-CAM was computed on the ResNet50 backbone's final convolutional
block (`layer4`) for four representative validation examples: a correct
`normal` prediction, a correct abnormal (`wheeze`) prediction, a
misclassification (true `crackle`, predicted `wheeze`), and a `both`-
class example. The `both`-class example is itself informative by its
failure: the model predicts `normal` for a true-`both` recording,
consistent with the class's F1 ≈ 0.18 — among the lowest of any
class/configuration combination in this thesis (Section 4.3) — and with
the class's small support (84 of 1,119 validation samples, the smallest
of the four classes), suggesting the model has too few `both` examples
to learn a distinct acoustic signature for the co-occurring-sounds
case, rather than reliably confusing it with something specific.

*[Figures: `results/figures/gradcam/gradcam_correct_normal.png`,
`gradcam_correct_abnormal.png`, `gradcam_incorrect.png`,
`gradcam_both_any.png` — insert with captions describing what region of
the spectrogram the heatmap highlights for each case.]*

## 4.9 Uncertainty: MC Dropout

30 stochastic forward passes per validation sample (dropout active at
inference, BatchNorm statistics frozen — Section 3.6), batched into a
single forward call per sample for efficiency:

| | Mean predictive entropy |
|---|---|
| Correct predictions | 0.527 |
| Incorrect predictions | **0.645** |

Incorrect predictions carry meaningfully higher entropy than correct
ones, so the model's uncertainty is informative rather than
decorative — a necessary condition for the "the model can say 'I'm not
sure'" framing this thesis uses to justify why uncertainty estimation
matters for a screening tool (Section 1.3, Contributions). Per-class
mean entropy tracks the same difficulty ordering seen throughout this
chapter: `wheeze` (0.554) < `normal` (0.571) < `crackle` (0.585) <
`both` (0.631) — the model is least confident on exactly the class it
gets wrong most often.

## 4.10 Ablation summary

Consolidating Sections 4.3–4.9 into a single view of what each design
choice cost or bought, relative to the tuned ResNet50 reference:

| Factor | Effect on macro F1 | Effect on cross-domain | Effect on interpretability/uncertainty |
|---|---|---|---|
| Weighted sampler → class-weighted loss | −0.032 | not tested | not tested |
| + SpecAugment | −0.035 (but +accuracy) | not tested | not tested |
| Architecture (EfficientNet-B3 vs. ResNet50) | +0.006 (not significant, Section 4.5) | not tested on EfficientNet-B3 | not tested on EfficientNet-B3 |
| Zero-shot BRACETS transfer | n/a | AUROC 0.509 (chance) | — |
| Zero-shot SPRSound transfer | n/a | Macro F1 0.187 (worse) | — |
| MC Dropout | n/a | n/a | Entropy gap 0.527 vs. 0.645 (informative) |

## 4.11 Discussion

Three patterns recur across every experiment in this chapter regardless
of architecture, tuning configuration, or evaluation dataset:

1. **The `both` class is the system's structural weak point.** Lowest
   F1 in every CNN configuration (Section 4.3), the example the model
   gets wrong in the Grad-CAM analysis (Section 4.8), and the class with
   the highest predictive uncertainty (Section 4.9). This is consistent
   with it having the smallest support in the training data (84 of
   1,119 validation samples) rather than being an inherently harder
   acoustic pattern.
2. **Single-run comparisons are not trustworthy on this dataset.** The
   multi-seed spread (Section 4.5) exceeds the architecture gap it would
   otherwise be used to explain, and the AudioMAE reproducibility issue
   (Section 4.4.1) is a direct, if unplanned, demonstration of the same
   point at the level of an entire training run rather than just a
   seed.
3. **In-domain performance says nothing about cross-domain
   performance.** A model with respectable in-domain macro F1 (0.458)
   collapses to nothing distinguishable from chance (BRACETS) or worse
   (SPRSound) the moment the recording environment changes — the
   clearest evidence in this thesis that "80% accuracy on ICBHI" and
   "clinically useful" are not the same claim.
