import sys
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from training.ai_dataset import create_dataloaders
from training.training_model import UIFAFTrainingModel


CHECKPOINT = Path(
    "outputs/checkpoints/ai_generation_best.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():

    print("======================================")
    print("UIFAF AI-GENERATION EVALUATION")
    print("======================================")

    print("Device:", DEVICE)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    _, _, test_loader = create_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    print("Test samples:", len(test_loader.dataset))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UIFAFTrainingModel(
        pretrained=True
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Best validation accuracy:",
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------
    # Test
    # --------------------------------------------------

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            # AI experiment:
            # 0 = Authentic
            # 1 = AI Generated
            predictions = torch.argmax(
                logits[:, :2],
                dim=1
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

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
        all_predictions,
        labels=[0, 1]
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("======================================")
    print("TEST RESULTS")
    print("======================================")

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1-score  : {f1 * 100:.2f}%"
    )

    print()
    print("Confusion Matrix")
    print("-----------------")
    print("                Predicted")
    print("              Real     AI")
    print(
        f"Actual Real   {cm[0][0]:4d}     {cm[0][1]:4d}"
    )
    print(
        f"Actual AI     {cm[1][0]:4d}     {cm[1][1]:4d}"
    )

    print()
    print("Classification Report")
    print("----------------------")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=[
                "Authentic",
                "AI Generated",
            ],
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Save confusion matrix
    # --------------------------------------------------

    output_dir = Path(
        "outputs/evaluation/ai_generation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cm_path = (
        output_dir /
        "confusion_matrix.png"
    )

    plt.figure(figsize=(6, 5))

    plt.imshow(cm)

    plt.title(
        "UIFAF - AI Generation Confusion Matrix"
    )

    plt.xticks(
        [0, 1],
        ["Authentic", "AI Generated"]
    )

    plt.yticks(
        [0, 1],
        ["Authentic", "AI Generated"]
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
                va="center",
            )

    plt.tight_layout()

    plt.savefig(
        cm_path,
        dpi=200
    )

    plt.close()

    print(
        "Confusion matrix saved:",
        cm_path
    )

    print()
    print("======================================")
    print("EVALUATION COMPLETE")
    print("======================================")


if __name__ == "__main__":
    main()