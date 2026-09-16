# Data Setup

## Почему это отдельный документ

Сырые датасеты (~12 ГБ суммарно) намеренно не в git — `data/raw/datasets`
в `.gitignore`. При переносе проекта на новую машину (или клонировании
репозитория) их нужно разместить или засимлинковать заново.

## Ожидаемая структура

`configs/data_config.yaml` ожидает такую структуру внутри
`data/raw/datasets/`:

```
data/raw/datasets/
├── archive/
│   └── Respiratory_Sound_Database/
│       └── Respiratory_Sound_Database/
│           ├── audio_and_txt_files/      # ICBHI 2017
│           └── patient_diagnosis.csv
├── HF_LUNG_V1/
│   ├── train/
│   └── test/
├── jwyy9np4gv-3/                         # KAUH
│   ├── Audio Files/
│   └── Data annotation.xlsx
├── BRACETS/
│   ├── Data/
│   └── Metadata/Metadata.txt
└── SPRSound-main/
    └── Classification/
        ├── train_classification_wav/
        ├── train_classification_json/
        ├── valid_classification_wav/
        └── valid_classification_json/
```

## Как разместить данные

`data/raw/datasets` может быть обычной директорией с данными **или**
символической ссылкой на директорию с такой же внутренней структурой —
парсеры (`src/data/parsers/*.py`) работают одинаково в обоих случаях.

Пример (если данные лежат в другом месте на диске):

```bash
ln -s "/absolute/path/to/your/datasets" data/raw/datasets
```

**Важно:** такой symlink — локальная настройка машины, не коммитить его
в git (уже в `.gitignore`; раньше по ошибке был закоммичен с абсолютным
путём конкретного Mac — если видишь это в старой истории git, это была
известная и исправленная проблема, не шаблон для подражания).

## Быстрая проверка, что всё на месте

```bash
MPLCONFIGDIR=results/.mplconfig venv/bin/python -m pytest tests/data/ -v
```

Если данные не найдены, тесты, помеченные `@pytest.mark.slow`,
пропускаются автоматически (см. `tests/conftest.py::data_available`) —
это ожидаемо в CI/окружении без датасетов, а не ошибка.
