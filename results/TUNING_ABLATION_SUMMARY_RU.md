# Tuning / Ablation Summary: ResNet50 on ICBHI

## Назначение

Три финальных прогона ResNet50 (10 epochs, early stopping + scheduler
во всех трёх) отличаются друг от друга ровно одним изменением
относительно tuned-конфигурации — сводит их в одну таблицу, вместо трёх
разрозненных логов в `results/.mplconfig/`.

Источники: `log_final_resnet50_tuned.txt`,
`log_final_resnet50_class_weighted_no_sampler.txt`,
`log_final_resnet50_specaugment.txt`.

## Конфигурации

| Конфигурация | Weighted sampler | Class-weighted loss | SpecAugment | Изменение относительно tuned |
|---|---|---|---|---|
| `tuned` (референс) | ON | OFF | OFF | — |
| `class_weighted_no_sampler` | OFF | ON | OFF | sampler → loss (гипотеза из PROJECT_LOG 2026-05-04) |
| `specaugment` | ON | OFF | ON | + augmentation поверх tuned |

## Результаты (best epoch по val_loss)

| Конфигурация | best epoch | val_loss | val_acc | val macro F1 |
|---|---|---|---|---|
| `tuned` | 1 | 1.1935 | 0.5407 | **0.4617** |
| `class_weighted_no_sampler` | 2 | 1.1580 | 0.5076 | 0.4262 |
| `specaugment` | 1 | **1.0127** | **0.5541** | 0.4226 |

## Per-class TP (best val confusion matrix, diagonal)

| Конфигурация | normal | crackle | wheeze | both |
|---|---|---|---|---|
| `tuned` | 342 | 159 | 75 | 29 |
| `class_weighted_no_sampler` | 249 | 226 | 81 | 12 |
| `specaugment` | 391 | 171 | 47 | 11 |

## Честный вывод

1. **Weighted sampler сам по себе сильнее, чем class-weighted loss без
   sampler.** Гипотеза из `PROJECT_LOG_RU.md` (2026-05-04) — «отключить
   sampler, оставить только loss, чтобы увидеть его чистый эффект» —
   подтверждена экспериментом, но с обратным результатом: даже в чистом
   виде class-weighted loss без sampler уступает tuned-конфигурации
   (`macro F1` 0.4262 против 0.4617). Практический вывод для методологии:
   в этом проекте выгоднее балансировать данные на уровне сэмплирования,
   а не на уровне функции потерь.
2. **SpecAugment улучшает accuracy и val_loss, но ухудшает macro F1** —
   и это не противоречие, а понятный эффект: аугментация вместе с уже
   включённым sampler'ом смещает модель в сторону доминирующих классов
   (`normal`: 342→391, `crackle`: 159→171), но резко проседает на редких
   (`wheeze`: 75→47, `both`: 29→11). Это ровно тот случай, когда
   `accuracy` вводит в заблуждение, а `macro F1` — нет: как раз то, о чём
   явно предупреждает дизайн-спек проекта.
3. Класс `both` остаётся системно самым слабым результатом во всех трёх
   конфигурациях без исключения — это не случайность одного прогона, а
   устойчивый паттерн, достойный отдельного абзаца в Discussion.
