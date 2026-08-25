# Историчный дневник работ по проекту

## Правило ведения

Это один из главных файлов проекта.

Назначение:

- фиксировать **что именно было сделано** по дипломке;
- хранить **хронологию решений**;
- сохранять **экспериментальные результаты и переходы между этапами**;
- не терять контекст, почему мы приняли те или иные технические решения.

Главное правило:

- старые записи **не переписываются заново**;
- новые события **добавляются ниже**;
- если понимание или решение изменилось, это оформляется **новой записью**, а не стиранием истории.

Этот файл дополняет:

- `PROGRESS.md` — текущее состояние и мастер-план;
- `thesis/STUDY_JOURNAL_RU.md` — что изучили и поняли;
- `results/BASELINE_SUMMARY_RU.md` — сводка baseline-экспериментов.

---

## 2026-04-01 — Старт проекта и формализация темы

### Что зафиксировали

- тема дипломки: автоматический анализ респираторных звуков с помощью глубокого обучения;
- общая цель: не просто одна модель, а полный pipeline от аудио до интерпретируемого результата;
- базовая дорожная карта:
  - data pipeline
  - baseline CNN
  - transformer
  - explainability
  - uncertainty
  - demo
  - thesis writing

### Почему это было важно

На этом шаге стало ясно, что дипломка должна быть:

- и инженерной;
- и исследовательской;
- и объяснимой для защиты.

---

## 2026-04-01 — 2026-04-14 — Сборка data pipeline

### Что сделали

- собрали и проверили parsers для:
  - `ICBHI`
  - `HF Lung`
  - `KAUH`
  - `BRACETS`
  - `SPRSound`
- приняли unified label scheme:
  - `0 = normal`
  - `1 = crackle`
  - `2 = wheeze`
  - `3 = both`
- реализовали:
  - `preprocessing.py`
  - `RespDataset`
  - `patient-level split`
  - `DataLoader factory`
  - weighted sampler

### Что стало важным решением

- использовать `patient-level split`, чтобы не допустить data leakage;
- приводить сегменты к фиксированному размеру окна;
- вынести production-код в `src/`, а не смешивать его с тестами и черновиками.

---

## 2026-04-15 — 2026-04-30 — Очистка структуры проекта

### Что сделали

- закрепили рабочий Python-код под `src/`;
- вынесли запускные сценарии в `scripts/`;
- перенесли конфиг в `configs/data_config.yaml`;
- почистили лишние артефакты и дублирующий код;
- обновили `.gitignore`.

### Почему это было важно

Без чистой структуры проект быстро превращается в набор случайных файлов.  
На этом шаге репозиторий стал похож на нормальный ML-проект, а не на набор черновиков.

---

## 2026-05-03 — Сборка первого рабочего training pipeline

### Что сделали

В `src/train.py` собрали baseline training utilities:

- `TrainConfig`
- `TrainHistory`
- `train_one_epoch()`
- `validate()`
- `build_optimizer()`
- `build_criterion()`
- `run_baseline()`
- `run_cnn_experiment()`
- `run_cnn_experiment_with_split()`
- `run_icbhi_baseline()`
- `train_model()`

### Что это дало

Появился первый полноценный baseline pipeline:

- parser -> split -> dataloader -> model -> train -> validation -> checkpoint

---

## 2026-05-03 — Исправление загрузки сегментов

### Что сделали

В `RespDataset` исправили логику:

- вместо чтения всегда с начала файла
- стали использовать `start` как `offset` для сегментно-размеченных записей

### Почему это было критично

До исправления модель фактически училась не на реальных размеченных дыхательных сегментах, а на начале записи.  
После исправления обучение стало методологически корректнее.

---

## 2026-05-03 — Первый воспроизводимый entrypoint экспериментов

### Что сделали

Создан:

- `scripts/train_icbhi_baseline.py`

### Что он умеет

- запускать baseline CNN на `ICBHI`;
- выбирать модель через CLI;
- сохранять checkpoint;
- сохранять learning curves;
- сохранять confusion matrix;
- писать summary в CSV.

### Почему это важно

Это был переход от “набора функций” к **воспроизводимому эксперименту**.

---

## 2026-05-03 — Проверка локального GPU на MacBook Air M1

### Что сделали

- выяснили, что `cuda` на Mac недоступна;
- добавили поддержку `mps`;
- проверили `torch.backends.mps.is_available() == True`;
- подтвердили, что baseline training реально идет на `mps`.

### Почему это важно

Это резко улучшило локальную практичность экспериментов по сравнению с CPU-only запуском.

---

## 2026-05-03 — Первый debug-run на ICBHI

### Результат

`ResNet50`, `1 epoch`, `mps`:

- `train_loss = 1.0388`
- `train_acc = 0.5551`
- `val_loss = 1.1110`
- `val_acc = 0.5290`

### Что это показало

- pipeline живой;
- checkpoint сохраняется;
- данные, модель и валидация работают end-to-end.

---

## 2026-05-03 — Полные baseline-runs для двух CNN

### Что сделали

Прогнали:

- `ResNet50`
- `EfficientNet-B3`

на `ICBHI` по `5 epochs`.

### Что получили

`ResNet50`:

- `best_epoch = 1`
- `best_val_loss = 1.1757`
- `final_val_acc = 0.5862`
- `final_val_macro_f1 = 0.4379`

`EfficientNet-B3`:

- `best_epoch = 2`
- `best_val_loss = 1.1251`
- `final_val_acc = 0.5424`
- `final_val_macro_f1 = 0.4338`

### Главный вывод

- обе CNN-модели рано начали переобучаться;
- `ResNet50` оказался немного сильнее по итоговой accuracy и macro F1;
- `EfficientNet-B3` дал лучший `best_val_loss`, но общая картина overfitting сохранилась.

---

## 2026-05-03 — Автоматическая фиксация baseline-артефактов

### Что сделали

Скрипт baseline-обучения научили автоматически сохранять:

- `results/baseline_results.csv`
- plots кривых обучения
- plots confusion matrix
- отдельные checkpoints по моделям

### Артефакты

- `results/figures/icbhi_resnet50_best_curves.png`
- `results/figures/icbhi_resnet50_best_confusion_matrix.png`
- `results/figures/icbhi_efficientnet_b3_best_curves.png`
- `results/figures/icbhi_efficientnet_b3_best_confusion_matrix.png`
- `results/BASELINE_SUMMARY_RU.md`

---

## 2026-05-03 — Теоретическая база проекта

### Что сделали

Создали большой теоретико-практический конспект:

- `thesis/THESIS_DEEP_DIVE_RU.md`

Потом собрали его в PDF:

- `thesis/THESIS_DEEP_DIVE_RU.pdf`

### Что внутри

- медицина;
- аудиообработка;
- CNN;
- ResNet50;
- EfficientNet-B3;
- AST;
- метрики;
- explainability;
- uncertainty;
- вопросы для защиты.

---

## 2026-05-04 — Запуск историчных журналов

### Что сделали

Созданы:

- `thesis/STUDY_JOURNAL_RU.md`
- `thesis/PROJECT_LOG_RU.md`

### Новое правило проекта

Теперь важные изменения должны фиксироваться в трех слоях:

1. `PROGRESS.md` — что сейчас готово и что дальше делать;
2. `STUDY_JOURNAL_RU.md` — что изучили и поняли;
3. `PROJECT_LOG_RU.md` — что именно сделали по проекту и в какой исторической последовательности.

---

## 2026-05-04 — Первый regularization-блок: scheduler + early stopping

### Что сделали

В `src/train.py` добавили:

- `scheduler_patience`
- `scheduler_factor`
- `early_stopping_patience`
- `ReduceLROnPlateau`
- `early stopping`

В `scripts/train_icbhi_baseline.py` добавили соответствующие CLI-аргументы.

### Эксперимент

`ResNet50` с:

- `epochs = 10`
- `early_stopping_patience = 2`
- `scheduler_patience = 1`
- `scheduler_factor = 0.5`

### Результат

- `Best epoch = 2`
- `Best val loss = 1.1144`
- `Final val acc = 0.5639`
- `Final val macro F1 = 0.4728`
- `Early stopping triggered at epoch 4`

### Главный вывод

- новый regularization-блок улучшил baseline;
- `best_val_loss` стал лучше старого baseline;
- `macro F1` вырос;
- переобучение не исчезло, но стало заметно лучше контролироваться.

### Следующий шаг

- добавить `class-weighted loss`;
- повторить `ResNet50`;
- затем при необходимости перенести лучшую схему на `EfficientNet-B3`.

---

## 2026-05-04 — Первый тест class-weighted loss

### Что сделали

В training pipeline добавили поддержку `class-weighted loss`:

- в `TrainConfig` добавлен флаг `use_class_weights`;
- реализован `compute_class_weights(...)`;
- `build_criterion(...)` научен использовать `nn.CrossEntropyLoss(weight=...)`;
- CLI-скрипт `train_icbhi_baseline.py` получил флаг `--use-class-weights`.

### Что протестировали

Запущен `ResNet50` с конфигурацией:

- `early stopping`
- `scheduler`
- `class-weighted loss`
- при этом `weighted sampler` остался включенным

### Результат эксперимента

- `Epoch 1`
  - `val_loss = 1.4090`
  - `val_acc = 0.3119`
  - `val_macro_f1 = 0.2674`
- `Epoch 2`
  - `val_loss = 1.2562`
  - `val_acc = 0.4182`
  - `val_macro_f1 = 0.3587`
- `Epoch 3`
  - `val_loss = 1.3696`
  - `val_acc = 0.4155`
  - `val_macro_f1 = 0.3532`
- `Epoch 4`
  - `val_loss = 1.3905`
  - `val_acc = 0.5290`
  - `val_macro_f1 = 0.4395`
- `Best epoch = 2`
- `Best val loss = 1.2562`
- `Final val acc = 0.5290`
- `Final val macro F1 = 0.4395`
- `Early stopping triggered at epoch 4`
- final validation confusion matrix:
  - `[266, 200, 71, 23]`
  - `[68, 223, 11, 9]`
  - `[28, 24, 94, 18]`
  - `[7, 23, 45, 9]`

### Вывод

Эта конфигурация в итоге все равно выглядит хуже, чем предыдущий improved baseline без `class-weighted loss`.

Наиболее вероятная причина:

- одновременно используются
  - `weighted sampler`
  - и `class-weighted loss`

Это, вероятно, слишком агрессивно усиливает редкие классы и ухудшает общую validation-динамику.

При этом эксперимент не является полностью провальным:

- к `epoch 4` `val_macro_f1` частично восстановился до `0.4395`;
- однако это все равно ниже, чем у конфигурации `early stopping + scheduler` без class weights (`0.4728`);
- лучший `val_loss` (`1.2562`) также заметно хуже предыдущего improved baseline (`1.1144`).

### Следующий шаг

Следующий чистый эксперимент:

- **отключить `weighted sampler`**
- **оставить `class-weighted loss`**

Так можно будет изолировать влияние loss и понять, полезен ли он сам по себе.
