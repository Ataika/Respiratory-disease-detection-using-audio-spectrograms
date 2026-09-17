# Chapter 2 — Related Work

## 2.1 The ICBHI 2017 benchmark

The ICBHI 2017 Respiratory Sound Database [R0] established the standard
benchmark this thesis builds on: patient-recorded lung sounds annotated
at the respiratory-cycle level with crackle/wheeze presence, released
alongside a public challenge and an official scoring convention —
Sensitivity, Specificity, and Score = (Se+Sp)/2 on the binary
normal-vs-abnormal collapse of the classification. That scoring
convention is why this thesis reports the same metric in Chapter 4
alongside macro F1, rather than only the latter.

> **[VERIFY]** Full citation to confirm before submission:
> B. M. Rocha et al., "A Respiratory Sound Database for the Development
> of Automated Classification," in *Precision Medicine Powered by
> pHealth and Connected Health (ICBHI 2017)*, IFMBE Proceedings, 2018.
> (Author list and exact proceedings title should be checked against the
> official ICBHI dataset page before the bibliography is finalized —
> this entry was reconstructed from memory, not re-verified via a fresh
> search this session.)

## 2.2 CNN-based approaches

Convolutional networks applied to mel-spectrogram representations of
lung sounds have been the dominant approach on ICBHI since shortly after
the dataset's release, typically using ImageNet-pretrained backbones
(ResNet [He et al., 2016], EfficientNet [Tan and Le, 2019]) fine-tuned on
spectrogram "images." Polanco-Martagón et al. [R2] compare five such CNN
backbones (VGG16/19, ResNet152V2, InceptionV3, MobileNetV3-Large)
specifically for adventitious lung-sound classification, and — notably
for this thesis's own contribution — explicitly flag as a limitation of
their study that results come from "a single training run for each
optimal configuration, not using a formal statistical analysis of
performance variance," citing computational cost as the reason. This
thesis's multi-seed robustness protocol (Section 4.5) directly addresses
that gap for the ResNet50/EfficientNet-B3 comparison, on the same
dataset and metric family.

## 2.3 Transformer and self-supervised approaches

The Audio Spectrogram Transformer (AST) [Gong et al., 2021] brought
supervised-pretrained transformers to audio classification, and Bae et
al. [R3] apply it directly to ICBHI with a patch-mix contrastive
learning objective, reporting an ICBHI Score of 62.37% — the closest
published precedent for this thesis's transformer branch. This thesis
originally planned to use AST directly; the substitution made instead
(AudioMAE, a self-supervised masked-autoencoder-pretrained ViT) and the
reasoning behind it are detailed in Section 3.6.1. Jeong and Kim [R5]
extend this line of work with a BEATs backbone and patient-level
contrastive losses specifically designed to prevent a model from
learning per-patient acoustic signatures instead of pathology signal —
independent validation, from a different angle, of why this thesis uses
a strict patient-level train/validation split (Section 3.8) rather than
a simple random split.

## 2.4 Cross-domain generalization

Most published ICBHI work, including the studies above, reports
in-domain performance only. Kim et al. [R1] are a direct exception:
they study distribution shift explicitly caused by recording-equipment
differences, treating each stethoscope type used to capture ICBHI as a
distinct domain, and apply supervised contrastive learning to adapt
across them — improving ICBHI Score by 2.16 percentage points over an
unadapted baseline. Their framing — that respiratory-sound classifiers
absorb equipment- and protocol-specific acoustic signatures alongside,
or instead of, clinically meaningful ones — is the closest published
account of the mechanism behind this thesis's own cross-domain
experiments (Section 4.6), which test transfer to two entirely separate
datasets (BRACETS, SPRSound) rather than different stethoscopes within
ICBHI itself — a coarser but more clinically realistic domain shift.

## 2.5 Explainability and calibration

Saky et al. [R6] combine Grad-CAM, Integrated Gradients, and SHAP with
an explicit "domain-shift evaluation" as a first-class part of their
study design for lung-disease detection from respiratory audio — the
closest structural analog to this thesis's combination of
interpretability (Grad-CAM, Section 4.8), predictive uncertainty (MC
Dropout, Section 4.9), and cross-domain evaluation in a single
evaluation protocol. Reliability and calibration receive comparatively
little attention in the ICBHI literature; this thesis's use of Expected
Calibration Error (ECE) and MC Dropout entropy (Sections 3.8, 4.9) is
intended to close part of that gap specifically for the medical-screening
use case, where a well-calibrated "I'm not sure" is arguably as
important as raw accuracy.

## 2.6 Reporting conventions

Fraihi et al. [R4] report results as mean ± standard deviation across
five independent runs on both ICBHI and SPRSound, the closest published
match to this thesis's own multi-seed reporting convention (Section
4.5, Chapter 4's results table). Their study, like most others reviewed
here, evaluates ICBHI and SPRSound with separately tuned models rather
than testing zero-shot transfer between them — this thesis's contrast
between in-domain and zero-shot cross-dataset performance on the exact
same trained model (Section 4.6, 4.7) makes that comparison explicit
rather than implicit.

## 2.7 Gap statement

Individually, prior work has addressed multi-seed variance reporting
[R2, R4], cross-domain adaptation within a single dataset's equipment
variation [R1], patient-level generalization [R5], and explainability
alongside domain-shift evaluation [R6]. Despite this progress, no study
we are aware of jointly reports seed-level variance *and* cross-dataset
zero-shot transfer for the same set of architectures — a combination
this thesis provides, evaluated across three architecture families
(two supervised CNNs, one self-supervised transformer) and two fully
independent unseen datasets.

---

## Bibliography entries for this chapter (IEEE style)

[R0] B. M. Rocha et al., "A Respiratory Sound Database for the
Development of Automated Classification," in *ICBHI 2017*, 2018.
**[VERIFY before submission — see note in 2.1]**

[R1] J.-W. Kim, S. Bae, W.-Y. Cho, B. Lee, and H.-Y. Jung,
"Stethoscope-guided Supervised Contrastive Learning for Cross-domain
Adaptation on Respiratory Sound Classification," in *Proc. IEEE Int.
Conf. Acoustics, Speech and Signal Processing (ICASSP)*, 2024.

[R2] S. Polanco-Martagón, Y. Hernández-Mier, M. A. Nuño-Maganda,
J. H. Barrón-Zambrano, A. Magadán-Salazar, and C. A. Medellín-Vergara,
"Comparison of Deep Neural Networks for the Classification of
Adventitious Lung Sounds," *J. Clinical Medicine*, vol. 14, no. 20,
p. 7427, 2025.

[R3] S. Bae, J.-W. Kim, W.-Y. Cho, H. Baek, S. Son, B. Lee, C. Ha,
K. Tae, S. Kim, and S.-Y. Yun, "Patch-Mix Contrastive Learning with
Audio Spectrogram Transformer on Respiratory Sound Classification," in
*Proc. INTERSPEECH*, 2023.

[R4] N. Fraihi, O. Karrakchou, and M. Ghogho, "Improving Deep
Learning-based Respiratory Sound Analysis with Frequency Selection and
Attention Mechanism," in *Proc. IEEE Eng. Medicine and Biology Conf.
(EMBC)*, 2025. arXiv:2507.20052.

[R5] S. G. Jeong and S. E. Kim, "Patient-Aware Feature Alignment for
Robust Lung Sound Classification: Cohesion-Separation and Global
Alignment Losses," in *Proc. INTERSPEECH*, 2025. arXiv:2505.23834.

[R6] S. M. A. I. Saky, M. R. Islam, M. S. Arefin, and S. Alam,
"Explainable Multi-Modal Deep Learning for Automatic Detection of Lung
Diseases from Respiratory Audio Signals," arXiv:2512.00563, 2025.

*(He et al. 2016 [ResNet], Tan & Le 2019 [EfficientNet], and Gong et
al. 2021 [AST] are canonical architecture papers cited by name above —
add their full entries when compiling the final bibliography; omitted
here since they were not part of this session's literature search and
should be pulled from the standard citation for each, not reconstructed
from memory.)*
