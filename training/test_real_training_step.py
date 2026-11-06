import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from training.ai_dataset import create_dataloaders
from training.training_model import UIFAFTrainingModel


def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    # --------------------------------------------------
    # DataLoader
    # --------------------------------------------------

    train_loader, _, _ = create_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    print("Images:", images.shape)
    print("Labels:", labels.tolist())

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = UIFAFTrainingModel(
        pretrained=True
    ).to(device)

    model.train()

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4,
        weight_decay=1e-4,
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------

    optimizer.zero_grad(set_to_none=True)

    logits = model(images)

    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    loss = criterion(logits, labels)

    print("Loss:", loss.item())

    # --------------------------------------------------
    # Backward
    # --------------------------------------------------

    loss.backward()

    print("Backward: SUCCESS")

    # --------------------------------------------------
    # Check gradients
    # --------------------------------------------------

    gradient_found = False

    for name, parameter in model.named_parameters():

        if (
            parameter.requires_grad
            and parameter.grad is not None
        ):

            if not torch.isfinite(
                parameter.grad
            ).all():

                raise RuntimeError(
                    f"Invalid gradient: {name}"
                )

            gradient_found = True

            print(
                "Gradient OK:",
                name,
                parameter.grad.norm().item()
            )

            break

    if not gradient_found:
        raise RuntimeError(
            "No trainable parameter received a gradient."
        )

    # --------------------------------------------------
    # Optimizer step
    # --------------------------------------------------

    optimizer.step()

    print("Optimizer step: SUCCESS")

    print()
    print("======================================")
    print("REAL TRAINING STEP PASSED")
    print("======================================")


if __name__ == "__main__":
    main()