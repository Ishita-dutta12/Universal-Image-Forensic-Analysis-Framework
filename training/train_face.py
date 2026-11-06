from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.celeb_dataset import create_dataloaders
from training.training_model import UIFAFTrainingModel


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BATCH_SIZE = 2
GRADIENT_ACCUMULATION = 4

MAX_EPOCHS = 10
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4

PATIENCE = 3

CHECKPOINT_DIR = Path("outputs/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR / "face_manipulation_best.pt"
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model, loader, device, criterion):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=device.type == "cuda"
            ):

                logits = model(images)

                binary_logits = logits[:, [0, 1]]

                loss = criterion(
                    binary_logits,
                    labels
                )

            total_loss += loss.item() * labels.size(0)

            predictions = binary_logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("======================================")
    print("UIFAF FACE MANIPULATION TRAINING")
    print("======================================")

    print("Device:", device)

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # -----------------------------------------------------
    # Data
    # -----------------------------------------------------

    train_loader, val_loader, _ = create_dataloaders(
        root="datasets/CelebHQ-FM",
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print("\nTrain batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    model = UIFAFTrainingModel(
        pretrained=True
    ).to(device)

    # -----------------------------------------------------
    # Loss
    # -----------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -----------------------------------------------------
    # Mixed precision
    # -----------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=device.type == "cuda"
    )

    # -----------------------------------------------------
    # Training state
    # -----------------------------------------------------

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for batch_idx, (images, labels) in enumerate(
            train_loader
        ):

            images = images.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=device.type == "cuda"
            ):

                logits = model(images)

                # Binary face manipulation experiment
                binary_logits = logits[:, [0, 1]]

                loss = criterion(
                    binary_logits,
                    labels
                )

                loss_for_backward = (
                    loss / GRADIENT_ACCUMULATION
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            # ---------------------------------------------
            # Statistics
            # ---------------------------------------------

            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = binary_logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            # ---------------------------------------------
            # Gradient accumulation
            # ---------------------------------------------

            if (
                (batch_idx + 1)
                % GRADIENT_ACCUMULATION == 0
            ):

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

        # ---------------------------------------------
        # Handle remaining gradients
        # ---------------------------------------------

        if len(train_loader) % GRADIENT_ACCUMULATION != 0:

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        # ---------------------------------------------
        # Train metrics
        # ---------------------------------------------

        train_loss = running_loss / total
        train_accuracy = correct / total

        # ---------------------------------------------
        # Validation
        # ---------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            device,
            criterion
        )

        print("\n--------------------------------------")
        print(
            f"Epoch {epoch}/{MAX_EPOCHS}"
        )
        print(
            f"Train Loss      : {train_loss:.4f}"
        )
        print(
            f"Train Accuracy  : {train_accuracy * 100:.2f}%"
        )
        print(
            f"Val Loss        : {val_loss:.4f}"
        )
        print(
            f"Val Accuracy    : {val_accuracy * 100:.2f}%"
        )
        print(
            f"Learning Rate   : "
            f"{optimizer.param_groups[0]['lr']:.7f}"
        )

        if torch.cuda.is_available():

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"GPU Memory      : "
                f"{allocated:.2f} GB allocated / "
                f"{reserved:.2f} GB reserved"
            )

        # ---------------------------------------------
        # Save best model
        # ---------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                CHECKPOINT_PATH,
            )

            print(
                "✓ Best checkpoint saved."
            )

        else:

            epochs_without_improvement += 1

            print(
                f"No validation improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

            if (
                epochs_without_improvement
                >= PATIENCE
            ):

                print(
                    "\nEarly stopping triggered."
                )

                break

    print("\n======================================")
    print("TRAINING COMPLETE")
    print("======================================")

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )


if __name__ == "__main__":
    main()