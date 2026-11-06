from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.training_model import UIFAFTrainingModel
from training.imd_dataset import IMD2020CombinedDataset


# ============================================================
# Configuration
# ============================================================

TRAIN_MANIFEST = Path("datasets/IMD2020/train.csv")
VAL_MANIFEST = Path("datasets/IMD2020/val.csv")

CHECKPOINT_DIR = Path(
    "outputs/checkpoints/imd2020"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BATCH_SIZE = 2

EPOCHS = 15

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

LOCALIZATION_WEIGHT = 1.0

PATIENCE = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Dice Loss
# ============================================================

def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:

    probabilities = torch.sigmoid(logits)

    probabilities = probabilities.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        probabilities * targets
    ).sum(dim=1)

    denominator = (
        probabilities.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return 1.0 - dice.mean()


# ============================================================
# Dice Score
# ============================================================

def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        predictions * targets
    ).sum(dim=1)

    denominator = (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
    )

    dice = (
        2.0 * intersection + 1.0
    ) / (
        denominator + 1.0
    )

    return dice.mean().item()


# ============================================================
# IoU Score
# ============================================================

def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:

    probabilities = torch.sigmoid(logits)

    predictions = (
        probabilities >= 0.5
    ).float()

    predictions = predictions.flatten(1)
    targets = targets.flatten(1)

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + 1.0
    ) / (
        union + 1.0
    )

    return iou.mean().item()


# ============================================================
# Binary classification metrics
# ============================================================

def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
):
    predictions = torch.argmax(
        logits,
        dim=1
    )

    tp = (
        (predictions == 1)
        & (targets == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        & (targets == 0)
    ).sum().item()

    fp = (
        (predictions == 1)
        & (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        & (targets == 1)
    ).sum().item()

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2.0 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        tp,
        tn,
        fp,
        fn,
    )


# ============================================================
# Build binary classical logits
# ============================================================

def build_classical_logits(
    logits: torch.Tensor,
) -> torch.Tensor:

    authentic = logits[:, 0]

    classical = torch.logsumexp(
        logits[:, 5:8],
        dim=1
    )

    return torch.stack(
        [
            authentic,
            classical,
        ],
        dim=1
    )


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    classification_loss_fn,
    localization_loss_fn,
):

    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    total_correct = 0
    total_samples = 0

    for images, masks, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        logits, localization_logits = model(
            images,
            return_localization=True
        )

        classical_logits = build_classical_logits(
            logits
        )

        # ----------------------------------------------------
        # Classification loss
        # ----------------------------------------------------

        cls_loss = classification_loss_fn(
            classical_logits,
            labels
        )

        # ----------------------------------------------------
        # Localization loss
        # ----------------------------------------------------

        bce = localization_loss_fn(
            localization_logits,
            masks
        )

        d_loss = dice_loss(
            localization_logits,
            masks
        )

        loc_loss = bce + d_loss

        # ----------------------------------------------------
        # Total loss
        # ----------------------------------------------------

        loss = (
            cls_loss
            + LOCALIZATION_WEIGHT * loc_loss
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        total_loss += loss.item()

        total_cls_loss += (
            cls_loss.item()
        )

        total_loc_loss += (
            loc_loss.item()
        )

        predictions = torch.argmax(
            classical_logits,
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    batches = len(loader)

    accuracy = (
        total_correct / total_samples
        if total_samples > 0
        else 0.0
    )

    return (
        total_loss / batches,
        total_cls_loss / batches,
        total_loc_loss / batches,
        accuracy,
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    classification_loss_fn,
    localization_loss_fn,
):

    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_loc_loss = 0.0

    all_logits = []
    all_labels = []

    total_dice = 0.0
    total_iou = 0.0

    for images, masks, labels in loader:

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        masks = masks.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        logits, localization_logits = model(
            images,
            return_localization=True
        )

        classical_logits = build_classical_logits(
            logits
        )

        cls_loss = classification_loss_fn(
            classical_logits,
            labels
        )

        bce = localization_loss_fn(
            localization_logits,
            masks
        )

        d_loss = dice_loss(
            localization_logits,
            masks
        )

        loc_loss = bce + d_loss

        loss = (
            cls_loss
            + LOCALIZATION_WEIGHT * loc_loss
        )

        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_loc_loss += loc_loss.item()

        all_logits.append(
            classical_logits.cpu()
        )

        all_labels.append(
            labels.cpu()
        )

        batch_size = labels.size(0)

        total_dice += (
            dice_score(
                localization_logits,
                masks
            )
            * batch_size
        )

        total_iou += (
            iou_score(
                localization_logits,
                masks
            )
            * batch_size
        )

    all_logits = torch.cat(
        all_logits,
        dim=0
    )

    all_labels = torch.cat(
        all_labels,
        dim=0
    )

    (
        accuracy,
        precision,
        recall,
        f1,
        tp,
        tn,
        fp,
        fn,
    ) = classification_metrics(
        all_logits,
        all_labels
    )

    samples = len(all_labels)

    batches = len(loader)

    return {
        "loss": total_loss / batches,
        "classification_loss": (
            total_cls_loss / batches
        ),
        "localization_loss": (
            total_loc_loss / batches
        ),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": total_dice / samples,
        "iou": total_iou / samples,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("======================================")
    print("UIFAF IMD2020 SUPERVISED TRAINING")
    print("======================================")

    print("Device:", DEVICE)

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = IMD2020CombinedDataset(
    TRAIN_MANIFEST,
    train=True
)

    val_dataset = IMD2020CombinedDataset(
    VAL_MANIFEST,
    train=False
)

    val_dataset = IMD2020CombinedDataset(
        VAL_MANIFEST,
        train=False
    )

    print(
        "Train samples:",
        len(train_dataset)
    )

    print(
        "Val samples  :",
        len(val_dataset)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = UIFAFTrainingModel(
        device=DEVICE,
        pretrained=True
    ).to(DEVICE)

    # --------------------------------------------------------
    # Losses
    # --------------------------------------------------------

    class_weights = torch.tensor(
    [1315 / 282, 1.0],
    dtype=torch.float32,
    device=DEVICE,
)

    classification_loss_fn = nn.CrossEntropyLoss(
    weight=class_weights
)

    localization_loss_fn = nn.BCEWithLogitsLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print(
        "Trainable parameters:",
        sum(
            parameter.numel()
            for parameter in trainable_parameters
        )
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_score = -1.0

    epochs_without_improvement = 0

    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        print()
        print("======================================")
        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print("======================================")

        train_results = train_one_epoch(
            model,
            train_loader,
            optimizer,
            classification_loss_fn,
            localization_loss_fn,
        )

        (
            train_loss,
            train_cls_loss,
            train_loc_loss,
            train_accuracy,
        ) = train_results

        val_results = validate(
            model,
            val_loader,
            classification_loss_fn,
            localization_loss_fn,
        )

        print(
            f"Train Loss          : {train_loss:.4f}"
        )

        print(
            f"Train Classification: {train_cls_loss:.4f}"
        )

        print(
            f"Train Localization  : {train_loc_loss:.4f}"
        )

        print(
            f"Train Accuracy      : "
            f"{train_accuracy * 100:.2f}%"
        )

        print()

        print(
            f"Val Loss            : "
            f"{val_results['loss']:.4f}"
        )

        print(
            f"Val Classification  : "
            f"{val_results['classification_loss']:.4f}"
        )

        print(
            f"Val Localization    : "
            f"{val_results['localization_loss']:.4f}"
        )

        print(
            f"Val Accuracy        : "
            f"{val_results['accuracy'] * 100:.2f}%"
        )

        print(
            f"Val Precision       : "
            f"{val_results['precision'] * 100:.2f}%"
        )

        print(
            f"Val Recall          : "
            f"{val_results['recall'] * 100:.2f}%"
        )

        print(
            f"Val F1              : "
            f"{val_results['f1'] * 100:.2f}%"
        )

        print(
            f"Val Dice            : "
            f"{val_results['dice']:.4f}"
        )

        print(
            f"Val IoU             : "
            f"{val_results['iou']:.4f}"
        )

        print()

        print(
            "Confusion Matrix:"
        )

        print(
            f"  TN: {val_results['tn']}"
            f"  FP: {val_results['fp']}"
        )

        print(
            f"  FN: {val_results['fn']}"
            f"  TP: {val_results['tp']}"
        )

        # ----------------------------------------------------
        # Model selection
        #
        # Average classification F1 and localization Dice.
        # ----------------------------------------------------

        validation_score = (
            val_results["f1"]
            + val_results["dice"]
        ) / 2.0

        print(
            f"\nValidation Score: "
            f"{validation_score:.4f}"
        )

        # ----------------------------------------------------
        # Save best checkpoint
        # ----------------------------------------------------

        if validation_score > best_score:

            best_score = validation_score

            epochs_without_improvement = 0

            checkpoint_path = (
                CHECKPOINT_DIR
                / "imd2020_best.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "validation_score":
                        validation_score,
                    "validation_metrics":
                        val_results,
                },
                checkpoint_path
            )

            print(
                "\nBest checkpoint saved:"
            )

            print(checkpoint_path)

        else:

            epochs_without_improvement += 1

            print(
                "\nNo validation improvement:"
                f" {epochs_without_improvement}"
                f"/{PATIENCE}"
            )

        # ----------------------------------------------------
        # GPU memory
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
                f"GPU Memory: "
                f"{allocated:.2f} GB allocated / "
                f"{reserved:.2f} GB reserved"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            break

    print()
    print("======================================")
    print("IMD2020 TRAINING COMPLETE")
    print("======================================")

    print(
        "Best validation score:",
        best_score
    )

    print(
        "Checkpoint:",
        CHECKPOINT_DIR
        / "imd2020_best.pt"
    )


if __name__ == "__main__":
    main()