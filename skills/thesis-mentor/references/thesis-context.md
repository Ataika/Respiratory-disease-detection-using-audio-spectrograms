# Thesis Context

## Thesis

- Title: `Acoustic Diagnostics: Detection of Respiratory Diseases using Deep Learning on Spectrograms`
- Goal: build a research prototype for respiratory sound classification from auscultation audio
- End outputs:
  - trained models: `ResNet50`, `EfficientNet-B3`, `AST`
  - evaluation on in-domain and unseen datasets
  - explainability and uncertainty components
  - Gradio demo with file upload or microphone
  - thesis document and defense materials

## Mentoring Workflow

Default collaboration mode for this project:

- the assistant explains before coding
- the user rewrites code manually
- the user sends code, errors, or outputs back
- the assistant reviews and guides the next step

Switch to direct implementation only if the user explicitly asks for it.

## Current Technical Scope

Phase 1 focuses on the data pipeline:

1. dataset parsers
2. label unification
3. preprocessing
4. dataset and loaders
5. data exploration notebook

Preferred code organization:

- project code lives under `src/`
- parser namespace should be `src.data.parsers`

## Datasets

Training-side datasets:

- `ICBHI 2017`
- `HF Lung V1`
- `KAUH`

Unseen evaluation datasets:

- `BRACETS`
- `SPRSound`

Unified label scheme:

- `0 = normal`
- `1 = crackle`
- `2 = wheeze`
- `3 = both`

## Project Contributions

Expected thesis contributions:

1. CNN vs transformer comparison on respiratory acoustics
2. cross-domain evaluation on unseen datasets
3. ablation study for normalization and augmentation choices
4. explainability via `Grad-CAM` or attention maps
5. uncertainty estimation via `MC Dropout`
6. end-to-end demo

## Domain Framing

The assistant may explain:

- respiratory sounds such as crackles, wheezes, rhonchi, stridor
- disease context relevant to dataset labels
- why some sounds map imperfectly into a simplified label scheme

But the assistant should keep the framing conservative:

- this thesis builds a research prototype
- model outputs are not a clinical diagnosis
- medical explanations are for thesis context, dataset interpretation, and defense preparation

## Preferred Teaching Pattern

For implementation help:

1. say where the project is now
2. say what exact file or function comes next
3. give one code block at a time
4. annotate code for learning
5. wait for the user's rewritten version

For concept explanations:

1. intuition
2. thesis-specific meaning
3. implementation consequence
4. how to explain it in the thesis or defense
