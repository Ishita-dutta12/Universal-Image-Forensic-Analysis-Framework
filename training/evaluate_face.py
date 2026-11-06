from pathlib import Path
import sys

import torch
import torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.celeb_dataset import create_dataloaders
from training.training_model import UIFAFTrainingModel


CHECKPOINT_PATH = Path(
    "outputs/checkpoints/face_manipulation_best.pt"
)

OUTPUT_DIR = Path(
    "outputs/evaluation/face_manipulation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("======================================")
    print("UIFAF FACE MANIPULATION EVALUATION")
    print("======================================")

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    # ---------------------------------------------------------
    # Test DataLoader
    # ---------------------------------------------------------

    _, _, test_loader = create_dataloaders(
        root="datasets/CelebHQ-FM",
        batch_size=2,
        num_workers=0,
    )

    print(
        "Test batches:",
        len(test_loader)
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = UIFAFTrainingModel(
        pretrained=True
    ).to(device)

    # ---------------------------------------------------------
    # Load best checkpoint
    # ---------------------------------------------------------

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded checkpoint from epoch:",
        checkpoint.get("epoch")
    )

    print(
        "Checkpoint validation accuracy:",
        f"{checkpoint.get('val_accuracy', 0) * 100:.2f}%"
    )

    model.eval()

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    all_predictions = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for images, labels in test_loader:

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

                # UIFAF classes 0 and 1
                # are used for this binary experiment.
                binary_logits = logits[:, [0, 1]]

                loss = criterion(
                    binary_logits,
                    labels
                )

            predictions = binary_logits.argmax(
                dim=1
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            total_samples += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    test_loss = (
        total_loss / total_samples
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print("\n======================================")
    print("TEST RESULTS")
    print("======================================")

    print(
        f"Test Loss  : {test_loss:.4f}"
    )

    print(
        f"Accuracy   : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision  : {precision * 100:.2f}%"
    )

    print(
        f"Recall     : {recall * 100:.2f}%"
    )

    print(
        f"F1-score   : {f1 * 100:.2f}%"
    )

    # ---------------------------------------------------------
    # Confusion Matrix
    # ---------------------------------------------------------

    print("\nConfusion Matrix")
    print(
        "              Predicted"
    )
    print(
        "              Real    Fake"
    )

    print(
        f"Actual Real   {cm[0][0]:5d}   {cm[0][1]:5d}"
    )

    print(
        f"Actual Fake   {cm[1][0]:5d}   {cm[1][1]:5d}"
    )

    # ---------------------------------------------------------
    # Classification Report
    # ---------------------------------------------------------

    print("\nClassification Report")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Authentic",
                "Face Manipulated"
            ],
            zero_division=0
        )
    )

    # ---------------------------------------------------------
    # Save confusion matrix
    # ---------------------------------------------------------

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Face Manipulation Confusion Matrix"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        ["Authentic", "Face Manipulated"]
    )

    plt.yticks(
        [0, 1],
        ["Authentic", "Face Manipulated"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    confusion_path = (
        OUTPUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path,
        dpi=200
    )

    plt.close()

    print(
        "\nConfusion matrix saved:",
        confusion_path
    )

    print("\n======================================")
    print("EVALUATION COMPLETE")
    print("======================================")


if __name__ == "__main__":
    main()