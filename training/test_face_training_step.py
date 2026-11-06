from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.face_dataset import create_dataloaders
from training.training_model import UIFAFTrainingModel


def main():

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_loader, _, _ = create_dataloaders(
        root="datasets/face_manipulation",
        batch_size=2,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    print("Images:", images.shape)
    print("Labels:", labels.tolist())

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model = UIFAFTrainingModel(
        pretrained=True
    ).to(device)

    model.train()

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------
    logits = model(images)

    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    # ---------------------------------------------------------
    # IMPORTANT
    #
    # UIFAF has 12 output classes.
    #
    # For this experiment:
    #
    # 0 = Authentic
    # 1 = AI Generated
    # 2 = AI Edited
    # 3 = Face Swap
    # 4 = Deepfake
    # ...
    #
    # But our face experiment is binary:
    #
    # 0 = Real
    # 1 = Fake
    #
    # Therefore we train only using logits 0 and 1.
    # ---------------------------------------------------------

    binary_logits = logits[:, [0, 1]]

    loss = criterion(
        binary_logits,
        labels
    )

    print("Binary logits:", binary_logits)
    print("Loss:", loss.item())

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------
    optimizer.zero_grad()

    loss.backward()

    print("Backward: SUCCESS")

    # ---------------------------------------------------------
    # Check gradient
    # ---------------------------------------------------------
    gradient_found = False

    for name, parameter in model.named_parameters():

        if parameter.requires_grad and parameter.grad is not None:

            if torch.isfinite(parameter.grad).all():

                print(
                    "Gradient OK:",
                    name,
                    parameter.grad.abs().mean().item()
                )

                gradient_found = True
                break

    assert gradient_found, "No valid gradient found."

    # ---------------------------------------------------------
    # Optimizer step
    # ---------------------------------------------------------
    optimizer.step()

    print("Optimizer step: SUCCESS")

    print("\nFACE TRAINING STEP PASSED")


if __name__ == "__main__":
    main()