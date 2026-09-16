"""Training utilities for respiratory sound classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.data.loaders import create_dataloader
from src.data.dataset import create_splits
from src.data.parsers.icbhi import parse_icbhi
from src.models.audio_ssl import build_audio_ssl_model
from src.models.cnn_baseline import build_cnn_model


@dataclass
class TrainConfig:
    """
    Minimal training configuration for baseline experiments.
    """

    device: str = "cpu"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 5
    log_every: int = 1
    checkpoint_path: str = "results/checkpoints/best_model.pth"
    num_classes: int = 4
    early_stopping_patience: int = 2
    scheduler_patience: int = 1
    scheduler_factor: float = 0.5
    use_class_weights: bool = False
    label_smoothing: float = 0.0
    use_spec_augment: bool = False
    freeze_backbone: bool = False


@dataclass
class TrainHistory:
    """
    Store epoch-level training and validation metrics.
    """

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    train_f1: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)
    val_confusion_matrices: list[list[list[int]]] = field(default_factory=list)
    final_val_targets: list[int] = field(default_factory=list)
    final_val_predictions: list[int] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    best_val_acc: float = 0.0
    best_val_f1: float = 0.0
    best_val_confusion_matrix: list[list[int]] = field(default_factory=list)
    best_val_targets: list[int] = field(default_factory=list)
    best_val_predictions: list[int] = field(default_factory=list)


def _compute_macro_f1(
    targets: list[int],
    predictions: list[int],
) -> float:
    """
    Compute macro F1 for a multi-class classification epoch.
    """
    return float(
        f1_score(
            targets,
            predictions,
            average="macro",
            zero_division=0,
        )
    )


def _compute_confusion_matrix(
    targets: list[int],
    predictions: list[int],
    num_classes: int,
) -> list[list[int]]:
    """
    Compute the confusion matrix with a fixed class order.
    """
    matrix = confusion_matrix(
        targets,
        predictions,
        labels=list(range(num_classes)),
    )
    return matrix.tolist()

def compute_class_weights(
        samples: list[dict],
        num_classes: int,
        device: str
) -> torch.Tensor:
    """Compute inverse-frequency class weights from
    training samples"""
    class_counts = torch.zeros(num_classes,
                               dtype=torch.float32)
    for sample in samples:
        label = sample["label"]
        class_counts[label] += 1

    class_weights = 1.0 / (class_counts + 1e-8)
    class_weights = class_weights / class_weights.sum() * num_classes

    return class_weights.to(device)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> tuple[float, float, float]:
    """
    Train the model for one epoch and return average loss, accuracy, and macro F1.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_targets: list[int] = []
    epoch_predictions: list[int] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(config.device)
        batch_y = batch_y.to(config.device)

        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == batch_y).sum().item()
        total_samples += batch_x.size(0)
        epoch_targets.extend(batch_y.detach().cpu().tolist())
        epoch_predictions.extend(predictions.detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError("Training dataloader is empty.")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    macro_f1 = _compute_macro_f1(epoch_targets, epoch_predictions)

    return avg_loss, avg_acc, macro_f1


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    config: TrainConfig,
) -> tuple[float, float, float, list[list[int]], list[int], list[int]]:
    """
    Evaluate the model on a validation set and return loss, accuracy,
    macro F1, confusion matrix, targets, and predictions.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_targets: list[int] = []
    epoch_predictions: list[int] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(config.device)
        batch_y = batch_y.to(config.device)

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item() * batch_x.size(0)

        predictions = logits.argmax(dim=1)
        total_correct += (predictions == batch_y).sum().item()
        total_samples += batch_x.size(0)
        epoch_targets.extend(batch_y.detach().cpu().tolist())
        epoch_predictions.extend(predictions.detach().cpu().tolist())

    if total_samples == 0:
        raise ValueError("Validation dataloader is empty.")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    macro_f1 = _compute_macro_f1(epoch_targets, epoch_predictions)
    matrix = _compute_confusion_matrix(
        epoch_targets,
        epoch_predictions,
        num_classes=config.num_classes,
    )

    return avg_loss, avg_acc, macro_f1, matrix, epoch_targets, epoch_predictions


def build_optimizer(
    model: nn.Module,
    config: TrainConfig,
) -> torch.optim.Optimizer:
    """
    Build the optimizer for baseline training.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """
    Build a learning rate scheduler that
     reduces the learning rate when validation
      loss stops improving
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min", ## reducing validation loss
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
    )


def build_criterion(
        config: TrainConfig,
        class_weights: torch.Tensor | None = None,
) -> nn.Module:
    """
    Build the loss function for respiratory sound classification.
    """
    if config.use_class_weights:
        if class_weights is None:
            raise ValueError("class_weights might be provided"
                             "when use_class_weights=True")
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.label_smoothing,
        )
    return nn.CrossEntropyLoss(
        label_smoothing=config.label_smoothing,
    )


def run_baseline(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: TrainConfig,
    class_weights: torch.Tensor | None = None,
) -> TrainHistory:
    """
    Build the baseline training components and run training.
    """
    if config.freeze_backbone:
        if not hasattr(model, "freeze_backbone"):
            raise ValueError(
                "freeze_backbone=True is not supported for this model."
            )
        model.freeze_backbone()

    model = model.to(config.device)

    criterion = build_criterion(
        config=config,
        class_weights=class_weights,
    )
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    history = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    return history


def build_experiment_model(
    model_name: str,
    num_classes: int,
) -> nn.Module:
    """
    Build a model for the current experiment family.
    """
    normalized_name = model_name.lower()

    if normalized_name in {"resnet50", "efficientnet_b3"}:
        return build_cnn_model(
            model_name=normalized_name,
            num_classes=num_classes,
        )

    if normalized_name == "audiomae":
        return build_audio_ssl_model(
            model_name=normalized_name,
            num_classes=num_classes,
        )

    raise ValueError(f"Unsupported experiment model: {model_name}")


def run_model_experiment(
    model_name: str,
    train_samples: list[dict],
    val_samples: list[dict],
    config: TrainConfig,
    batch_size: int = 8,
    branch: str = "cnn",
    sample_rate: int = 22050,
    duration: float = 5.0,
    use_weighted_sampler: bool = True,
) -> tuple[nn.Module, TrainHistory]:
    """
    Build a model, create dataloaders, and run training.
    """
    model = build_experiment_model(
        model_name=model_name,
        num_classes=config.num_classes,
    )

    train_dataloader = create_dataloader(
        samples=train_samples,
        branch=branch,
        sample_rate=sample_rate,
        duration=duration,
        batch_size=batch_size,
        shuffle=not use_weighted_sampler,
        use_weighted_sampler=use_weighted_sampler,
        use_spec_augment=config.use_spec_augment,
    )

    val_dataloader = create_dataloader(
        samples=val_samples,
        branch=branch,
        sample_rate=sample_rate,
        duration=duration,
        batch_size=batch_size,
        shuffle=False,
        use_weighted_sampler=False,
        use_spec_augment=False,
    )

    class_weights = None
    if config.use_class_weights:
        class_weights = compute_class_weights(
            samples=train_samples,
            num_classes=config.num_classes,
            device=config.device,
        )

    history = run_baseline(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        class_weights=class_weights,
    )

    return model, history


def run_model_experiment_with_split(
    model_name: str,
    samples: list[dict],
    config: TrainConfig,
    val_ratio: float = 0.15,
    seed: int = 42,
    batch_size: int = 8,
    branch: str = "cnn",
    sample_rate: int = 22050,
    duration: float = 5.0,
    use_weighted_sampler: bool = True,
) -> tuple[nn.Module, TrainHistory]:
    """
    Split samples at patient level and run a model experiment.
    """
    train_samples, val_samples = create_splits(
        samples=samples,
        val_ratio=val_ratio,
        seed=seed,
    )
    model, history = run_model_experiment(
        model_name=model_name,
        train_samples=train_samples,
        val_samples=val_samples,
        config=config,
        batch_size=batch_size,
        branch=branch,
        sample_rate=sample_rate,
        duration=duration,
        use_weighted_sampler=use_weighted_sampler,
    )

    return model, history


def run_icbhi_baseline(
    model_name: str,
    dataset_path: str,
    config: TrainConfig,
    val_ratio: float = 0.15,
    seed: int = 42,
    batch_size: int = 8,
    use_weighted_sampler: bool = True,
) -> tuple[nn.Module, TrainHistory]:
    """
    Parse the ICBHI dataset and run the ICBHI baseline experiment.
    """
    samples = parse_icbhi(dataset_path)

    branch = "cnn"
    sample_rate = 22050
    duration = 5.0

    if model_name.lower() == "audiomae":
        branch = "audiomae"
        sample_rate = 16000
        duration = 10.0

    model, history = run_model_experiment_with_split(
        model_name=model_name,
        samples=samples,
        config=config,
        val_ratio=val_ratio,
        seed=seed,
        batch_size=batch_size,
        branch=branch,
        sample_rate=sample_rate,
        duration=duration,
        use_weighted_sampler=use_weighted_sampler,
    )

    return model, history


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    config: TrainConfig,
) -> TrainHistory:
    """
    Run the full training loop across epochs, collect metrics,
    and save the best model based on validation loss.
    """
    history = TrainHistory()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.num_epochs):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
        )
        (
            val_loss,
            val_acc,
            val_f1,
            val_confusion_matrix,
            val_targets,
            val_predictions,
        ) = validate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            config=config,
        )

        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.train_f1.append(train_f1)
        history.val_f1.append(val_f1)
        history.val_confusion_matrices.append(val_confusion_matrix)
        history.final_val_targets = val_targets
        history.final_val_predictions = val_predictions

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history.best_val_loss = val_loss
            history.best_epoch = epoch + 1
            history.best_val_acc = val_acc
            history.best_val_f1 = val_f1
            history.best_val_confusion_matrix = val_confusion_matrix
            history.best_val_targets = val_targets
            history.best_val_predictions = val_predictions
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1

        epoch_number = epoch + 1
        should_log = (
            epoch_number == 1
            or epoch_number == config.num_epochs
            or epoch_number % config.log_every == 0
        )
        if should_log:
            print(
                f"Epoch {epoch_number}/{config.num_epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"train_f1={train_f1:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | "
                f"val_f1={val_f1:.4f}"
            )

        if epochs_without_improvement >= config.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch_number}. "
                f"No validation loss improvement for "
                f"{config.early_stopping_patience} consecutive epochs."
            )
            break

    return history
