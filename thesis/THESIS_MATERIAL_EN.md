# Thesis Material: Citations, Tables, and Drafted Paragraphs

Ready-to-paste English material for the thesis body, built from real project
numbers and from a literature pass finding directly comparable prior work.
Source of the numbers: `results/advanced_metrics/advanced_metrics_report.json`,
`results/multiseed_results.csv`, `results/external/*`. AudioMAE row/paragraphs
marked `[PENDING]` until its verified retrain finishes — do not fill in
numbers there from the earlier (quarantined, possibly race-corrupted)
checkpoint.

---

## 1. Related Work — bibliography entries (IEEE style, matches `source.txt`)

[R1] J.-W. Kim, S. Bae, W.-Y. Cho, B. Lee, and H.-Y. Jung, "Stethoscope-
guided Supervised Contrastive Learning for Cross-domain Adaptation on
Respiratory Sound Classification," in *Proc. IEEE Int. Conf. Acoustics,
Speech and Signal Processing (ICASSP)*, 2024.

[R2] S. Polanco-Martagón, Y. Hernández-Mier, M. A. Nuño-Maganda,
J. H. Barrón-Zambrano, A. Magadán-Salazar, and C. A. Medellín-Vergara,
"Comparison of Deep Neural Networks for the Classification of Adventitious
Lung Sounds," *J. Clinical Medicine*, vol. 14, no. 20, p. 7427, 2025.

[R3] S. Bae, J.-W. Kim, W.-Y. Cho, H. Baek, S. Son, B. Lee, C. Ha, K. Tae,
S. Kim, and S.-Y. Yun, "Patch-Mix Contrastive Learning with Audio
Spectrogram Transformer on Respiratory Sound Classification," in *Proc.
INTERSPEECH*, 2023.

[R4] N. Fraihi, O. Karrakchou, and M. Ghogho, "Improving Deep
Learning-based Respiratory Sound Analysis with Frequency Selection and
Attention Mechanism," in *Proc. IEEE Eng. Medicine and Biology Conf.
(EMBC)*, 2025. arXiv:2507.20052.

[R5] S. G. Jeong and S. E. Kim, "Patient-Aware Feature Alignment for
Robust Lung Sound Classification: Cohesion-Separation and Global
Alignment Losses," in *Proc. INTERSPEECH*, 2025. arXiv:2505.23834.

[R6] S. M. A. I. Saky, M. R. Islam, M. S. Arefin, and S. Alam, "Explainable
Multi-Modal Deep Learning for Automatic Detection of Lung Diseases from
Respiratory Audio Signals," arXiv:2512.00563, 2025.

**Where each earns its citation:**
- [R1] — cite when explaining *why* zero-shot BRACETS/SPRSound transfer
  collapses (equipment + protocol domain shift, not just population).
- [R2] — cite in Discussion right where you report the multi-seed spread;
  they name the absence of variance analysis as their own limitation.
- [R3] — cite when introducing the transformer branch / justifying the
  AST→AudioMAE substitution; direct precedent for pretrained-transformer-
  on-ICBHI.
- [R4] — cite when presenting the results table; matches your mean±SD
  reporting convention almost exactly.
- [R5] — cite alongside your patient-level split description; validates
  it as methodologically load-bearing, not just good practice.
- [R6] — cite opening the Interpretability chapter (Grad-CAM + uncertainty
  alongside domain-shift evaluation — structurally the closest analog to
  your combination).

---

## 2. Abstract — opening two sentences

> Automated classification of respiratory sounds from lung auscultation
> offers a low-cost route to early screening for diseases such as COPD and
> pneumonia, and recent deep learning models — from convolutional networks
> to self-supervised transformers — have pushed benchmark performance on
> the ICBHI 2017 dataset past previous state of the art. However, it
> remains unclear whether these gains reflect genuine acoustic
> understanding or dataset-specific overfitting: models are rarely
> evaluated across recording equipment, patient populations, or multiple
> random initializations, leaving their real-world reliability an open
> question.

---

## 3. Related Work — gap statement (closing paragraph)

> Despite this progress, no study we are aware of jointly reports
> seed-level variance and cross-dataset zero-shot transfer for the same
> set of architectures — a combination this thesis provides.

---

## 4. Methodology — opening paragraph

> This chapter describes how five heterogeneous respiratory-sound
> datasets were unified into a single labeling scheme and used to train
> three architectures under a protocol designed to answer three
> questions: (1) does a self-supervised transformer outperform supervised
> CNNs on ICBHI once seed variance is accounted for; (2) how much of that
> performance survives a shift in recording equipment and patient
> population; (3) which acoustic regions drive correct versus incorrect
> predictions.

---

## 5. Methodology — AST → AudioMAE substitution

> We initially selected the Audio Spectrogram Transformer (AST) as the
> transformer branch, following its established track record on ICBHI
> [R3]. We substituted AudioMAE — a self-supervised ViT pretrained via
> masked-autoencoding on AudioSet — because its pretraining objective does
> not depend on AudioSet's label taxonomy aligning with respiratory sound
> characteristics, unlike AST's supervised pretraining on sound-event
> tags. This keeps the comparison structurally analogous to prior
> CNN-vs-transformer ICBHI studies while testing a self-supervised
> pretraining signal that remains comparatively unexplored on this
> dataset relative to supervised AST fine-tuning.

*(Note: this is a defensible technical justification, not a historical
record of why the swap actually happened — the project's own journals
never documented the original reasoning. If you recall the real reason,
replace this paragraph; otherwise this one is honest and citable as-is.)*

---

## 6. Results table — field-standard format (Se / Sp / Score, mean where available)

Matches the reporting convention in [R4]: Sensitivity, Specificity,
ICBHI Score = (Se+Sp)/2, plus macro F1 (accuracy alone is not reported
as a headline metric — consistent with why the design spec required
macro F1 in the first place).

**Update, resolved**: this table and the note below it are superseded
by `thesis/chapters/04_experiments.md` Sections 4.4/4.4.1/4.5, which
have the corrected numbers (an ICBHI Score calculation bug was found
and fixed during QA — see that chapter for the full story) and the
completed AudioMAE multi-seed results. Kept here only for the drafted
citation-linked prose below, which is still current.

| Model | Se | Sp | ICBHI Score | Macro F1 (single run) | Macro F1 (5-seed mean ± SD) | AUC-ROC (macro OvR) | ECE |
|---|---|---|---|---|---|---|---|
| ResNet50 (tuned) | 0.589 | 0.752 | 0.670 | 0.458 | 0.403 ± 0.040 | 0.743 | 0.190 |
| EfficientNet-B3 (tuned) | 0.778 | 0.568 | 0.673 | 0.464 | 0.402 ± 0.028 | 0.777 | 0.114 |
| AudioMAE (no-sampler) | 0.283 | 0.805 | 0.544 | 0.235 | 0.334 ± 0.099 | 0.571 | 0.054 |

95% bootstrap CI (macro F1, single run): ResNet50 [0.423, 0.493];
EfficientNet-B3 [0.432, 0.496] — the two intervals overlap almost
entirely, consistent with the multi-seed finding below.

**Note the ICBHI Score vs macro F1 disagreement**: EfficientNet-B3 now
leads on *every* metric except calibration (ECE) — the earlier draft's
claim that ResNet50 "wins" on ICBHI Score was the bug mentioned above,
not a genuine finding. The real disagreement worth keeping in the text
is between the two CNNs' near-identical ICBHI Scores (0.670 vs 0.673,
effectively tied) despite very different Se/Sp balances — ResNet50
trades sensitivity for specificity, EfficientNet-B3 does the reverse,
and the composite metric hides that trade-off. That is itself the
argument for reporting the full table rather than a single headline
number, not the specific ranking claim the earlier draft made.

---

## 7. Discussion — cross-domain collapse (cite [R1])

> Zero-shot transfer to BRACETS yielded an AUROC of 0.509 (binary
> healthy-vs-pathological), and transfer to SPRSound yielded a macro F1
> of 0.187 — both indicating that the ResNet50 classifier's decision
> boundary, despite strong in-domain performance on ICBHI, carries little
> to no discriminative signal on an unseen recording setup. This is
> consistent with reports that respiratory-sound classifiers absorb
> equipment- and protocol-specific acoustic signatures rather than
> clinically generalizable ones [R1] — a phenomenon we quantify directly,
> across two independent unseen datasets, rather than infer indirectly.
> The SPRSound result is markedly worse than BRACETS, which we attribute
> to a compounding factor beyond equipment: SPRSound is a pediatric
> dataset, whereas ICBHI is predominantly adult, adding a population-level
> acoustic shift (airway geometry, respiratory rate) on top of the
> equipment shift both external sets share.

---

## 8. Discussion — multi-seed variance (cite [R2])

> Across five seeds, macro F1 for ResNet50 varied with SD ≈ 0.040 (mean
> 0.403, range 0.361–0.460) and for EfficientNet-B3 with SD ≈ 0.028 (mean
> 0.402, range 0.354–0.437) — a spread comparable to, or larger than, the
> mean gap between the two architectures in any single run (Δ = 0.001
> between means). Single-seed architecture comparisons remain common in
> the ICBHI literature; Polanco-Martagón et al. [R2], for instance,
> explicitly note the absence of a variance analysis as a limitation of
> their own cross-architecture comparison, citing computational cost.
> Our results suggest that, absent multi-seed reporting, such comparisons
> risk reversing model rankings by chance alone — the single-run ICBHI
> Score in Table [X] would rank ResNet50 above EfficientNet-B3 (0.580 vs
> 0.537), while the opposite is true for macro F1 (0.464 vs 0.458), and
> neither ranking is distinguishable from noise once seed variance is
> taken into account.

---

## 9. Limitations — template for the zero-shot scope boundary

> The cross-domain evaluation is zero-shot: no fine-tuning or domain
> adaptation was applied to BRACETS or SPRSound. This isolates the raw
> transferability of ICBHI-trained representations but does not measure
> the ceiling achievable with even minimal target-domain adaptation,
> which prior work on stethoscope-guided contrastive learning [R1]
> suggests can recover much of the gap.

---

## 10. What's still needed before these can be finalized

- **Done**: AudioMAE's full 5-seed run completed (results in Chapter 4
  Sections 4.4/4.4.1/4.5). It resolved to a genuinely stronger finding
  than "pending" — AudioMAE is not just weaker on average but ~3x less
  stable across seeds than either CNN, with its worst seed collapsing to
  zero recall on two of four classes. This changes the thesis's
  headline claim (Section 1.3/Abstract need to reflect that the planned
  "transformer wins" result did not hold up — see Chapter 4.4.1 for the
  reframed contribution).
- Section 5's AST-to-AudioMAE justification paragraph is technically
  sound but not verified against the student's actual memory of the
  decision — flag this explicitly if asked in the defense, don't
  present it as historical fact (already flagged inline in Chapter 3
  itself, not just here).
- Chapter 1 (Introduction) contributions list and Chapter 5 (Conclusion)
  still need writing, now that this result is final rather than
  pending — see the writing plan for the recommended order.
