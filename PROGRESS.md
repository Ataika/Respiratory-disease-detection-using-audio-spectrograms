# Прогресс дипломной работы

## Проект
**Тема:** "Acoustic Diagnostics: Detection of Respiratory Diseases using Deep Learning on Spectrograms"
**Студент:** Atai Keneshbekov, Università degli Studi di Messina, L-31, 4th year part-time
**Руководитель:** Prof. Daniele Ravi
**Дедлайн:** Июнь 2026

---

## Режим работы с Claude
- Claude отправляет код с подробными комментариями (каждая строка объяснена)
- Пользователь **переписывает код вручную** (не копипаст) — учится по дороге
- Пользователь присылает свой код + вывод/ошибки
- Claude проверяет, объясняет ошибки, одобряет и двигается дальше
- При новых концепциях предлагать `/ml-mentor`

---

## Архитектура системы

### Конечный продукт
1. **Три обученные модели:** ResNet50, EfficientNet-B3, AST
2. **Живая демо:** Gradio + микрофон → класс + Grad-CAM + uncertainty
3. **Дипломная работа:** ~60-80 страниц, 5 глав

### Технический стек
- PyTorch, librosa, Gradio, wandb
- Grad-CAM (объяснимость), MC Dropout (uncertainty)
- GPU: Google Colab / Kaggle

### Датасеты
| Датасет | Записей | Роль | Метки |
|---------|---------|------|-------|
| ICBHI 2017 | 920 | train | normal/crackle/wheeze/both |
| HF Lung V1 | 9,765 | train | normal/wheeze/crackle/etc |
| KAUH | 336 | train | normal/abnormal |
| BRACETS | 3,291 | unseen test | binary: healthy/pathological |
| SPRSound | 3,554 | unseen test | multiclass |

### Unified label scheme
```
0 = normal
1 = crackle
2 = wheeze
3 = both (crackle + wheeze)
```
BRACETS: binary → {normal→0, crackle/wheeze/both→1}

### Pipeline архитектура
```
RAW AUDIO (.wav, variable length)
      ↓
Resampling: 22050Hz (CNN) / 16000Hz (AST)
      ↓
Instance Normalization (per-recording, zero mean unit variance)
      ↓
Fixed-length windowing: 5 сек окна, 50% overlap (2.5 сек шаг)
      ↓
    CNN Branch                    AST Branch
Mel Spectrogram              128-mel filterbank
(n_mels=128, hop=512)        (hop=160, ~500 frames)
ImageNet нормализация        AudioSet нормализация
→ tensor (3, 128, T)         → tensor (128, T)
SpecAugment                  SpecAugment
      ↓                            ↓
ResNet50 / EfficientNet-B3        AST
      ↓
Weighted Loss (борьба с дисбалансом)
      ↓
Predictions + MC Dropout (uncertainty)
      ↓
Grad-CAM visualization
      ↓
Gradio Demo (микрофон → результат)
```

---

## Структура файлов проекта
```
src/
├── data/
│   ├── parsers/
│   │   ├── __init__.py
│   │   ├── icbhi.py          ← Task 2
│   │   ├── hf_lung.py        ← Task 3
│   │   ├── kauh.py           ← Task 4
│   │   ├── bracets.py        ← Task 4
│   │   └── spr_sound.py      ← Task 4
│   ├── dataset.py            ← Task 6
│   ├── preprocessing.py      ← Task 5
│   └── loaders.py            ← Task 7
configs/
└── data_config.yaml          ← ✅ ГОТОВО
tests/
├── conftest.py               ← ✅ ГОТОВО
└── data/
    ├── test_parsers.py       ← Task 2-4
    ├── test_dataset.py       ← Task 6
    ├── test_preprocessing.py ← Task 5
    └── test_loaders.py       ← Task 7
notebooks/
└── 01_data_exploration.ipynb ← Task 8
```

---

## Детальный план по фазам

## Актуальный мастер-план от начала до конца

### Линии работы
1. Кодовая база: data pipeline → models → evaluation → demo
2. Текст диплома: писать параллельно, а не в самом конце
3. Материалы защиты: таблицы, графики, Grad-CAM, demo, slides

### Этап 0 — Foundation
- Исправить `tests/conftest.py`
- Исправить `configs/data_config.yaml`
- Активировать `venv` и установить минимальные зависимости
- Добиться чистого запуска `python -m pytest tests/ -v`

### Этап 1 — Data Pipeline
- `ICBHI` parser
- `HF Lung V1` parser
- `KAUH`, `BRACETS`, `SPRSound` parsers
- Unified label mapping
- `patient-level split`
- `preprocessing.py`: resampling, instance norm, windowing, spectrograms
- `SpecAugment`
- `RespDataset`
- `DataLoader factory`
- notebook для статистики и визуализаций

### Этап 2 — Baseline Models
- `ResNet50`
- `EfficientNet-B3`
- training loop
- validation
- checkpoints
- logging
- baseline metrics

### Этап 3 — Transformer
- `AST`
- fine-tuning
- честное сравнение `ResNet50 vs EfficientNet-B3 vs AST`

### Этап 4 — Robustness and Science Part
- ablation study
- cross-domain evaluation
- `BRACETS` binary evaluation
- `SPRSound` unseen evaluation
- calibration / ECE

### Этап 5 — Explainability and Uncertainty
- `Grad-CAM`
- AST attention visualization
- `MC Dropout`
- confident / uncertain prediction examples

### Этап 6 — Demo
- `Gradio`
- upload `.wav`
- microphone input
- output: class + confidence + uncertainty + heatmap

### Этап 7 — Thesis Writing
- Chapter 3 `Methodology`
- Chapter 4 `Experiments & Results`
- Chapter 2 `Related Work`
- Chapter 1 `Introduction`
- Chapter 5 `Conclusion`
- final abstract and formatting

### Этап 8 — Finalization and Defense
- final figures and tables
- clean repo structure
- best checkpoints
- slides
- defense rehearsal
- offline demo check

## Текущая точка

- Работаем в режиме mentorship: пользователь переписывает код вручную, я даю код с объяснениями
- Общий план утверждён и сохранён
- Фундамент тестовой среды приведён в рабочее состояние
- `ICBHI parser` реализован и покрыт минимальными unit-тестами
- `HF Lung V1 parser` реализован и проверен на реальном датасете (`52,444` фазовых сегмента)
- `KAUH parser` реализован и покрыт unit-тестами
- `BRACETS parser` реализован и покрыт unit-тестами
- `SPRSound parser` реализован, mapping уточнён по реальным меткам (`Normal`, `DAS`, `CAS`, `CAS & DAS`, `Poor Quality`) и покрыт unit-тестами
- Весь data-layer собран и протестирован: `28/28` тестов проходят
- `preprocessing.py` реализован: instance normalization, audio loading, padding, CNN/AST feature branches
- `RespDataset` и patient-level split реализованы
- `DataLoader factory` и weighted sampler реализованы
- Initial EDA notebook собран: таблица датасетов, распределение классов, примеры спектрограмм
- Структура проекта очищена: Python-код закреплён под `src/`, запускные сценарии вынесены в `scripts/`, конфиг перенесён в `configs/data_config.yaml`
- `src/train.py` доведён до рабочего baseline-pipeline:
  - `TrainConfig`, `TrainHistory`
  - `train_one_epoch()`, `validate()`
  - `build_optimizer()`, `build_criterion()`
  - `run_baseline()`, `run_cnn_experiment()`, `run_cnn_experiment_with_split()`
  - `run_icbhi_baseline()`
  - epoch-level logging для всех эпох
  - `macro F1`
  - validation confusion matrix
- `scripts/train_icbhi_baseline.py` создан как первый воспроизводимый entrypoint для baseline-обучения на `ICBHI`
- baseline entrypoint автоматически сохраняет:
  - learning curves в `results/figures/`
  - confusion matrix в `results/figures/`
  - сводку эксперимента в `results/baseline_results.csv`
- Собран главный русскоязычный теоретико-практический конспект по всей дипломке:
  - `thesis/THESIS_DEEP_DIVE_RU.md`
  - включает медицину, обработку сигнала, CNN, ResNet, EfficientNet, AST, метрики, overfitting, explainability, uncertainty и подготовку к защите
  - дополнен расширенной картой внешних источников, маршрутом погружения и учебным планом по уровням
  - собран в отдельный читаемый PDF: `thesis/THESIS_DEEP_DIVE_RU.pdf`
  - добавлен локальный генератор PDF без внешних зависимостей: `scripts/build_thesis_deep_dive_pdf.py`
- Запущен отдельный историчный дневник изучения дипломки:
  - `thesis/STUDY_JOURNAL_RU.md`
  - ведется как хронологический журнал понимания, изучения и экспериментальных выводов
- Запущен отдельный историчный дневник работ по проекту:
  - `thesis/PROJECT_LOG_RU.md`
  - ведется как хронологический журнал сделанных шагов, решений и экспериментов
  - это теперь одно из главных правил проекта: историю не переписывать, а дополнять новыми записями
- В `RespDataset` исправлена логика загрузки: для сегментно-размеченных датасетов используется `start` как `offset`, а не всегда начало файла
- `src/data/loaders.py` очищен от неправильной зависимости на `tests`
- Локальный baseline-run подтверждён end-to-end:
  - парсинг `ICBHI`
  - patient-level split
  - dataloaders
  - preprocessing
  - обучение `ResNet50`
  - валидация
  - сохранение checkpoint
- Apple GPU backend (`mps`) проверен и подтверждён как доступный в `venv`
- Debug-run на `ICBHI` через `mps` успешно завершён:
  - `Epoch 1/1`
  - `train_loss = 1.0388`
  - `train_acc = 0.5551`
  - `val_loss = 1.1110`
  - `val_acc = 0.5290`
- Полноценный baseline-run `ResNet50` на `ICBHI` (`5 epochs`) завершён на `mps`:
  - `Epoch 1/5`
    - `train_loss = 1.0802`
    - `train_acc = 0.5141`
    - `train_macro_f1 = 0.5126`
    - `val_loss = 1.1757`
    - `val_acc = 0.4844`
    - `val_macro_f1 = 0.4174`
  - `Epoch 5/5`
    - `train_loss = 0.2524`
    - `train_acc = 0.9111`
    - `train_macro_f1 = 0.9116`
    - `val_loss = 1.3718`
    - `val_acc = 0.5862`
    - `val_macro_f1 = 0.4379`
  - `best_epoch = 1`
  - `best_val_loss = 1.1757`
  - final validation confusion matrix:
    - `[456, 69, 29, 6]`
    - `[172, 125, 11, 3]`
    - `[53, 8, 66, 37]`
    - `[19, 16, 40, 9]`
  - checkpoint:
    - `results/checkpoints/icbhi_resnet50_best.pth`
- Полноценный baseline-run `EfficientNet-B3` на `ICBHI` (`5 epochs`) завершён на `mps`:
  - `Epoch 1/5`
    - `train_loss = 1.1910`
    - `train_acc = 0.4605`
    - `train_macro_f1 = 0.4604`
    - `val_loss = 1.1407`
    - `val_acc = 0.4996`
    - `val_macro_f1 = 0.4138`
  - `Epoch 5/5`
    - `train_loss = 0.3671`
    - `train_acc = 0.8649`
    - `train_macro_f1 = 0.8641`
    - `val_loss = 1.4543`
    - `val_acc = 0.5424`
    - `val_macro_f1 = 0.4338`
  - `best_epoch = 2`
  - `best_val_loss = 1.1251`
  - final validation confusion matrix:
    - `[374, 119, 55, 12]`
    - `[141, 141, 20, 9]`
    - `[54, 4, 81, 25]`
    - `[20, 10, 43, 11]`
  - checkpoint:
    - `results/checkpoints/icbhi_efficientnet_b3_best.pth`
- Научный вывод по `ResNet50 baseline`:
  - модель уверенно обучается на train-наборе
  - training accuracy и training macro F1 к 5-й эпохе становятся очень высокими
  - validation accuracy растёт умеренно, но validation macro F1 остаётся заметно ниже
  - лучшая эпоха по `val_loss` достигается уже на `epoch 1`
  - validation loss ухудшается при сильном росте train accuracy
  - это ранний и явный сигнал `overfitting`
  - baseline рабочий, но текущая конфигурация уже показывает ограниченную generalization и неравномерное качество по классам
- Научный вывод по `EfficientNet-B3 baseline`:
  - модель также уверенно обучается на train-наборе
  - лучшая эпоха по `val_loss` достигается рано, уже на `epoch 2`
  - по итоговой validation accuracy уступает `ResNet50`
  - итоговый `val_macro_f1` также немного ниже, чем у `ResNet50`
  - validation loss тоже растёт к концу обучения
  - признак `overfitting` сохраняется и для второй CNN-модели
- Первое baseline-сравнение на `ICBHI`:
  - `ResNet50`: `val_acc = 0.5862`, `val_loss = 1.3718`, `val_macro_f1 = 0.4379`
  - `EfficientNet-B3`: `val_acc = 0.5424`, `val_loss = 1.4543`, `val_macro_f1 = 0.4338`
  - по итоговой `val_acc` и `val_macro_f1` лучше выглядит `ResNet50`
  - по `best_val_loss` тоже лучше выглядит `EfficientNet-B3` не в финале, а только в ранней точке обучения (`epoch 2 = 1.1251`), однако и эта модель быстро уходит в overfitting
  - обе модели показывают общий паттерн: сильный рост train-метрик, ранний лучший `val_loss`, трудности на редких и смешанных классах
- Артефакты baseline-сравнения зафиксированы:
  - summary:
    - `results/BASELINE_SUMMARY_RU.md`
  - CSV:
    - `results/baseline_results.csv`
  - `ResNet50` curves:
    - `results/figures/icbhi_resnet50_best_curves.png`
  - `ResNet50` confusion matrix:
    - `results/figures/icbhi_resnet50_best_confusion_matrix.png`
  - `EfficientNet-B3` curves:
    - `results/figures/icbhi_efficientnet_b3_best_curves.png`
  - `EfficientNet-B3` confusion matrix:
    - `results/figures/icbhi_efficientnet_b3_best_confusion_matrix.png`
- Рекомендуемый следующий шаг после baseline:
  - не делать сразу широкий brute-force hyperparameter tuning
  - сначала закрепить baseline как научную отправную точку
  - затем провести небольшой hypothesis-driven tuning:
    - `early stopping`
    - `learning rate scheduler`
    - `class-weighted loss` или `focal loss`
    - аккуратные augmentation-эксперименты
  - после этого переходить к `AST` и cross-domain evaluation
- Первый тест `class-weighted loss` зафиксирован как предварительно неудачный в текущей конфигурации:
  - при одновременном использовании `weighted sampler` и `class-weighted loss` validation-метрики просели
  - финальный `weighted` run завершился с:
    - `best_epoch = 2`
    - `best_val_loss = 1.2562`
    - `final_val_acc = 0.5290`
    - `final_val_macro_f1 = 0.4395`
  - run частично восстановился к `epoch 4`, но все равно уступил improved baseline без class weights
  - следующая чистая гипотеза: оставить `class-weighted loss`, но отключить `weighted sampler`
- Checkpoint baseline-модели сохранён:
  - `results/checkpoints/icbhi_best_model.pth`
- Реальные EDA-объёмы:
  - `ICBHI`: `6898`
  - `HF_LUNG`: `52444`
  - `KAUH`: `336`
  - `BRACETS`: `3291`
  - `SPRSound`: `1772`
- Для локального macOS-окружения добавлен `readline.py` stub, потому что нативный `readline` вызывал segfault при старте `pytest`
- Следующий обязательный шаг: сделать компактный регуляризационный/tuning-блок для CNN baseline и после этого переходить к `AST`

### Фаза 1 — Data Pipeline (1–14 апреля) ✅ ЗАВЕРШЕНА

#### Tasks и статус:
- [x] **Task 1:** Конфиг + структура папок + conftest.py
- [x] **Task 2:** ICBHI Parser
- [x] **Task 3:** HF Lung V1 Parser
- [x] **Task 4:** KAUH + BRACETS + SPRSound Parsers
- [x] **Task 5:** Preprocessing (instance norm + spectrograms core)
- [x] **Task 6:** RespDataset + patient-level split
- [x] **Task 7:** DataLoader Factory
- [x] **Task 8:** Initial Data Exploration Notebook

### Фаза 2 — Baseline Models (14–28 апреля)
- ResNet50 + EfficientNet-B3 с pretrained весами
- Training loop + wandb логирование
- Метрики: ICBHI Score, F1, AUC-ROC, ECE
- Текущий статус:
  - [x] `ResNet50` baseline model
  - [x] `EfficientNet-B3` baseline model factory
  - [x] training loop
  - [x] validation loop
  - [x] patient-level split orchestration
  - [x] checkpoint saving
  - [x] `ICBHI` baseline entrypoint script
  - [x] debug-run на `mps`
  - [x] полноценный `ResNet50` run (`5 epochs`)
  - [x] epoch-level logging
  - [x] `macro F1`
  - [x] confusion matrix
  - [x] полноценный `EfficientNet-B3` run (`5 epochs`)
  - [x] автосохранение графиков
  - [x] CSV-таблица baseline-результатов
  - [ ] интерпретированная таблица baseline-результатов
  - [ ] расширенные метрики (`F1`, `AUC-ROC`, `ECE`, `ICBHI score`)
- Следующий практический шаг:
  - запустить обе модели еще раз с новым логированием, чтобы получить `macro F1`, confusion matrix и `best epoch` в итоговом выводе
  - затем собрать интерпретированную baseline-таблицу для главы `Experiments`
  - затем зафиксировать результаты в таблице для главы `Experiments`

### Фаза 3 — AST + Сравнение (28 апр – 12 мая)
- AST (Audio Spectrogram Transformer, pretrained AudioSet)
- Сравнительная таблица всех трёх моделей
- Statistical significance: McNemar's test, DeLong's test, Bootstrap CI

### Фаза 4 — Interpretability + Unseen Test (12–26 мая)
- Grad-CAM для CNN моделей
- Attention maps для AST
- MC Dropout (N=50): uncertainty estimation
- ECE калибровочные кривые
- Ablation study: с/без instance norm, с/без SpecAugment, с/без weighted loss
- Cross-domain eval: BRACETS (binary) + SPRSound
- Ensemble: average softmax всех трёх моделей

### Фаза 5 — Demo (26 мая – 9 июня)
- Gradio: загрузка файла ИЛИ микрофон (браузер → 44.1kHz → ресемплинг 22050Hz)
- Вывод: класс + уверенность + Grad-CAM тепловая карта

### Фаза 6 — Финализация (9–23 июня)
- Introduction + Abstract (EN + IT)
- Редактура всего текста диплома
- Сдача

---

## Структура диплома (5 глав)
1. **Introduction** — проблема, мотивация, contributions
2. **Related Work** — обзор литературы
3. **Methodology** — данные, preprocessing, архитектуры
4. **Experiments & Results** — сравнение, ablation, интерпретируемость
5. **Conclusion** — выводы, ограничения, будущая работа

## Contributions диплома
1. Сравнение CNN vs Transformer на respiratory acoustics
2. Cross-domain валидация на unseen datasets
3. Ablation study каждой техники
4. Grad-CAM + MC Dropout для медицинской объяснимости
5. End-to-end демо с микрофонным вводом

---

## Пути к данным (реальные на этом Mac)
```
ICBHI:    data/raw/datasets/archive/Respiratory_Sound_Database/Respiratory_Sound_Database/
          audio_and_txt_files/*.wav + *.txt (start end crackle wheeze)
          patient_diagnosis.csv (patient_id, diagnosis)

HF_LUNG:  data/raw/datasets/HF_LUNG_V1/train/*.wav + *_label.txt
          label format: "Wheeze 00:00:01.5 00:00:02.4"
          Папки train 2-10 и test 2-3 — дубликаты, пропускать

KAUH:     data/raw/datasets/jwyy9np4gv-3/Audio Files/*.wav
          data/raw/datasets/jwyy9np4gv-3/Data annotation.xlsx

BRACETS:  data/raw/datasets/BRACETS/Data/{patient_id}/Sound/{pos}/*.wav
          data/raw/datasets/BRACETS/Metadata/Metadata.txt
          (SubjectID, Diagnosis: Healthy/COPD/ILD/Asthma)

SPRSO:    data/raw/datasets/SPRSound-main/Classification/train_classification_wav/*.wav
          data/raw/datasets/SPRSound-main/Classification/train_classification_json/*.json
          json: {"record_annotation": "Normal"|"Wheeze"|"Crackle"|"Wheeze&Crackle"}
```

---

## Важные технические решения
| Решение | Зачем |
|---------|-------|
| Instance normalization | Записи из разных клиник/оборудования |
| Patient-level split | Предотвращает data leakage |
| Weighted loss | ICBHI ~60% normal — иначе модель не учится |
| MC Dropout | "Не уверен — обратитесь к врачу" |
| Grad-CAM | Объяснимость для комиссии |
| Unseen test | Честная cross-domain валидация |
| Ablation study | Доказывает что каждая техника помогает |

---

## Документы
- **Spec (дизайн):** `/Users/ataika/Desktop/My-assistant-claude/docs/superpowers/specs/2026-04-01-thesis-acoustic-diagnostics-design.md`
- **Plan (детальный):** `/Users/ataika/Desktop/My-assistant-claude/docs/superpowers/plans/2026-04-01-phase1-data-pipeline.md`
