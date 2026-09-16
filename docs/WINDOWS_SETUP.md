# Windows CUDA Setup

## Цель

Развернуть проект на Windows-ноутбуке с NVIDIA GPU так, чтобы baseline training запускался через `CUDA`, а не через `mps` или `cpu`.

## Что переносить

Переносить нужно:

- весь код проекта;
- папки `src/`, `scripts/`, `tests/`, `docs/`, `thesis/`, `configs/`;
- папку `data/raw/datasets/`, если на Windows тоже будут локальные датасеты;
- папку `results/`, если хочешь сохранить историю экспериментов и графики.

Не нужно переносить:

- `venv/`
- `__pycache__/`
- `.pytest_cache/`
- временные системные файлы

## Лучший практический способ

### Вариант A — рекомендованный

1. Создать приватный GitHub-репозиторий или использовать уже существующий.
2. Запушить туда код проекта.
3. На Windows сделать `git clone`.
4. Датасеты перенести отдельно через внешний SSD, флешку или облако.

Плюсы:

- чище история;
- проще обновлять код между Mac и Windows;
- проект не превращается в набор архивов.

### Вариант B — если хочешь быстрее без Git

1. Заархивировать корень проекта без `venv/`.
2. Скопировать архив на Windows.
3. Распаковать в удобную папку, например:

```text
C:\Projects\respiratory-disease-detection-using-audio-spectrograms
```

4. Датасеты перенести отдельно.

## Что ставить на Windows

1. Python
2. Git
3. Обновленный NVIDIA драйвер

Для первого запуска **не используй WSL**.  
Для твоего кейса проще и надежнее сначала развернуть все в нативном Windows `PowerShell`.

## Настройка окружения

Открой `PowerShell` в корне проекта и выполни:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Если `py -3.12` не сработает, можно использовать:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## Установка PyTorch с CUDA

Ставь `torch`, `torchvision`, `torchaudio` отдельно официальной командой PyTorch.

Для CUDA 12.4:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Установка остальных зависимостей проекта

После PyTorch:

```powershell
pip install -r requirements-windows-min.txt
```

## Проверка CUDA

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Ожидаемый результат:

- версия `torch`
- `True`
- название твоей видеокарты, например `NVIDIA GeForce RTX 4060`

## Проверка проекта

```powershell
python -m py_compile src/train.py scripts/train_icbhi_baseline.py
```

## Первый test-run

```powershell
python scripts/train_icbhi_baseline.py --model-name resnet50 --epochs 1
```

В конце запуска должно быть:

```text
Device: cuda
```

## Полезный первый полноценный запуск

```powershell
python scripts/train_icbhi_baseline.py `
  --model-name resnet50 `
  --epochs 10 `
  --checkpoint-path results/checkpoints/icbhi_resnet50_cuda_best.pth `
  --early-stopping-patience 2 `
  --scheduler-patience 1 `
  --scheduler-factor 0.5
```

## Если хочешь перенести и датасеты без правки путей

Сохрани на Windows такую же относительную структуру:

```text
data/raw/datasets/archive/Respiratory_Sound_Database/Respiratory_Sound_Database
```

Тогда текущий `--dataset-path` в проекте почти не придется менять.

## Что делать, если `torch.cuda.is_available()` == `False`

Проверь по порядку:

1. установлен ли NVIDIA драйвер;
2. точно ли ты в `venv`;
3. не поставил ли ты CPU-версию PyTorch;
4. действительно ли поставил wheel из `cu124`;
5. не используешь ли случайно системный Python вместо `venv`.

## Что делать после первого успешного запуска

1. Повторить baseline runs на `CUDA`;
2. сохранить графики и CSV;
3. потом уже продолжать tuning и `AST`.
