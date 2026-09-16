# Baseline Summary: ICBHI CNN Models

## Назначение

Этот файл фиксирует первый полный baseline-comparison на датасете `ICBHI` для двух CNN-моделей:

- `ResNet50`
- `EfficientNet-B3`

Эксперименты запускались:

- на устройстве `mps`
- с `5 epochs`
- `batch_size = 8`
- `learning_rate = 1e-4`
- `weight_decay = 1e-4`
- `val_ratio = 0.15`
- `seed = 42`

## Итоговая таблица

| Model | Best Epoch | Best Val Loss | Final Train Acc | Final Train Macro F1 | Final Val Acc | Final Val Macro F1 | Checkpoint |
|------|------------|---------------|-----------------|----------------------|---------------|--------------------|------------|
| `ResNet50` | `1` | `1.1757` | `0.9111` | `0.9116` | `0.5862` | `0.4379` | `results/checkpoints/icbhi_resnet50_best.pth` |
| `EfficientNet-B3` | `2` | `1.1251` | `0.8649` | `0.8641` | `0.5424` | `0.4338` | `results/checkpoints/icbhi_efficientnet_b3_best.pth` |

## Главные выводы

### ResNet50

- по итоговой `validation accuracy` выглядит лучше;
- по итоговой `validation macro F1` тоже немного лучше;
- лучший `val_loss` достигается очень рано, уже на `epoch 1`;
- к `epoch 5` наблюдается явный `overfitting`.

### EfficientNet-B3

- лучшая эпоха по `val_loss` достигается на `epoch 2`;
- по итоговым метрикам немного уступает `ResNet50`;
- общий паттерн похожий: train-метрики растут сильно, а validation начинает деградировать;
- класс `both` и часть `crackle` сегментов остаются трудными.

### Общий baseline-вывод

- обе CNN-модели обучаются и извлекают полезные признаки из respiratory spectrograms;
- обе модели рано переобучаются;
- редкие и смешанные классы остаются проблемными;
- одной `accuracy` недостаточно, поэтому `macro F1` и confusion matrix обязательны для честного анализа.

## Артефакты экспериментов

### ResNet50

- Curves plot:
  [icbhi_resnet50_best_curves.png](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/figures/icbhi_resnet50_best_curves.png)
- Confusion matrix plot:
  [icbhi_resnet50_best_confusion_matrix.png](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/figures/icbhi_resnet50_best_confusion_matrix.png)
- Checkpoint:
  [icbhi_resnet50_best.pth](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/checkpoints/icbhi_resnet50_best.pth)

### EfficientNet-B3

- Curves plot:
  [icbhi_efficientnet_b3_best_curves.png](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/figures/icbhi_efficientnet_b3_best_curves.png)
- Confusion matrix plot:
  [icbhi_efficientnet_b3_best_confusion_matrix.png](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/figures/icbhi_efficientnet_b3_best_confusion_matrix.png)
- Checkpoint:
  [icbhi_efficientnet_b3_best.pth](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/checkpoints/icbhi_efficientnet_b3_best.pth)

### Таблица результатов CSV

- [baseline_results.csv](/Users/ataika/Desktop/Thesis%20Diploma/Respiratory-disease-detection-using-audio-spectrograms/results/baseline_results.csv)

## Confusion matrix values

### ResNet50

```text
[456, 69, 29, 6]
[172, 125, 11, 3]
[53, 8, 66, 37]
[19, 16, 40, 9]
```

### EfficientNet-B3

```text
[374, 119, 55, 12]
[141, 141, 20, 9]
[54, 4, 81, 25]
[20, 10, 43, 11]
```

## Что делать дальше

На этом этапе правильнее не уходить сразу в большой хаотичный hyperparameter sweep.

Лучший следующий путь:

1. Зафиксировать baseline как отправную точку и анализировать модели по `best epoch`, а не только по последней эпохе.
2. Добавить `per-class precision/recall/F1`, чтобы понять, где именно проваливаются `crackle`, `wheeze` и `both`.
3. Провести маленький гипотезно-управляемый tuning:
   - `early stopping`
   - `learning rate scheduler`
   - `class-weighted loss` или `focal loss`
   - аккуратные `augmentations`
4. После этого запускать `AST`, чтобы сравнение `CNN vs Transformer` было уже не сырым, а научно осмысленным.

То есть сначала:

- закрепить baseline;
- сделать точечный regularization/tuning;
- затем переходить к `AST` и внешней валидации.
