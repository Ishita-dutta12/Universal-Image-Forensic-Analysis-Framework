import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from training.ai_dataset import create_dataloaders
from training.training_model import UIFAFTrainingModel


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4

MAX_EPOCHS = 15
PATIENCE = 3

LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = Path(
    "outputs/checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda")
            ):

                logits = model(images)

                loss = criterion(
                    logits[:, :2],
                    labels
                )

            batch_size = labels.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

            predictions = torch.argmax(
                logits[:, :2],
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        correct / total_samples
    )

    return average_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("======================================")
    print("UIFAF AI-GENERATION TRAINING")
    print("======================================")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    print("\nLoading dataset...")

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
            num_workers=0,
        )
    )

    print(
        "Train samples:",
        len(train_loader.dataset)
    )

    print(
        "Validation samples:",
        len(val_loader.dataset)
    )

    print(
        "Test samples:",
        len(test_loader.dataset)
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print("\nLoading UIFAF model...")

    model = UIFAFTrainingModel(
        pretrained=True
    ).to(DEVICE)

    model.train()

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    trainable_parameters = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # LEARNING RATE SCHEDULER
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # MIXED PRECISION
    # --------------------------------------------------------

    scaler = GradScaler(
        "cuda",
        enabled=(DEVICE.type == "cuda")
    )

    # --------------------------------------------------------
    # BEST MODEL / EARLY STOPPING
    # --------------------------------------------------------

    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(1, MAX_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        running_correct = 0
        running_samples = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        for step, (images, labels) in enumerate(
            train_loader,
            start=1
        ):

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            with autocast(
                device_type="cuda",
                enabled=(DEVICE.type == "cuda")
            ):

                logits = model(images)

                # --------------------------------------------
                # AI experiment
                #
                # 0 = Authentic
                # 1 = AI Generated
                # --------------------------------------------

                loss = criterion(
                    logits[:, :2],
                    labels
                )

                scaled_loss = (
                    loss /
                    GRADIENT_ACCUMULATION_STEPS
                )

            scaler.scale(
                scaled_loss
            ).backward()

            batch_size = labels.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            running_samples += batch_size

            predictions = torch.argmax(
                logits[:, :2],
                dim=1
            )

            running_correct += (
                predictions == labels
            ).sum().item()

            # --------------------------------------------
            # Optimizer update
            # --------------------------------------------

            if (
                step %
                GRADIENT_ACCUMULATION_STEPS == 0
                or
                step == len(train_loader)
            ):

                scaler.step(optimizer)

                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

        # ----------------------------------------------------
        # TRAIN METRICS
        # ----------------------------------------------------

        train_loss = (
            running_loss /
            running_samples
        )

        train_accuracy = (
            running_correct /
            running_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion
        )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        scheduler.step(val_accuracy)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # PRINT
        # ----------------------------------------------------

        print()
        print(
            f"Epoch {epoch}/{MAX_EPOCHS}"
        )

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Train Accuracy  : "
            f"{train_accuracy * 100:.2f}%"
        )

        print(
            f"Val Loss        : {val_loss:.4f}"
        )

        print(
            f"Val Accuracy    : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Learning Rate   : {current_lr:.7f}"
        )

        # ----------------------------------------------------
        # BEST CHECKPOINT
        # ----------------------------------------------------

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            epochs_without_improvement = 0

            checkpoint_path = (
                CHECKPOINT_DIR /
                "ai_generation_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_accuracy": val_accuracy,
                },
                checkpoint_path
            )

            print(
                "Best model saved:",
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1

            print(
                "No validation improvement:",
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # ----------------------------------------------------
        # GPU MEMORY
        # ----------------------------------------------------

        if DEVICE.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated()
                / (1024 ** 3)
            )

            reserved = (
                torch.cuda.memory_reserved()
                / (1024 ** 3)
            )

            print(
                f"GPU Memory       : "
                f"{allocated:.2f} GB allocated / "
                f"{reserved:.2f} GB reserved"
            )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if epochs_without_improvement >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()
    print("======================================")
    print("TRAINING COMPLETE")
    print("======================================")

    print(
        f"Best validation accuracy: "
        f"{best_val_accuracy * 100:.2f}%"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_DIR /
        "ai_generation_best.pt"
    )


if __name__ == "__main__":
    main()