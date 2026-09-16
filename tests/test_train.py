import torch
import torch.nn as nn

import src.train as train_module
from src.train import TrainConfig, train_model


def test_train_model_tracks_best_validation_predictions(monkeypatch, tmp_path):
    """
    The exported per-class report must match the best checkpoint, not only the
    final epoch.
    """
    checkpoint_path = tmp_path / "best_model.pth"
    config = TrainConfig(
        device="cpu",
        num_epochs=2,
        checkpoint_path=str(checkpoint_path),
        early_stopping_patience=10,
    )
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    criterion = nn.CrossEntropyLoss()

    validation_results = iter(
        [
            (
                0.8,
                0.75,
                0.70,
                [[1, 0], [1, 0]],
                [0, 1],
                [0, 0],
            ),
            (
                0.9,
                0.50,
                0.40,
                [[0, 1], [0, 1]],
                [0, 1],
                [1, 1],
            ),
        ]
    )

    def fake_train_one_epoch(**_kwargs):
        return 0.5, 0.5, 0.5

    def fake_validate(**_kwargs):
        return next(validation_results)

    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_module, "validate", fake_validate)

    history = train_model(
        model=model,
        train_dataloader=object(),
        val_dataloader=object(),
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )

    assert history.best_epoch == 1
    assert history.best_val_loss == 0.8
    assert history.best_val_acc == 0.75
    assert history.best_val_f1 == 0.70
    assert history.best_val_confusion_matrix == [[1, 0], [1, 0]]
    assert history.best_val_targets == [0, 1]
    assert history.best_val_predictions == [0, 0]
    assert history.final_val_predictions == [1, 1]
    assert checkpoint_path.exists()
