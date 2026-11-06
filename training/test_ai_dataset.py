import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from training.ai_dataset import create_dataloaders


def main():

    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=2,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print("========== AI DATASET TEST ==========")

    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))

    print()
    print("Image shape   :", images.shape)
    print("Labels shape  :", labels.shape)
    print("Labels        :", labels.tolist())
    print("Image dtype   :", images.dtype)
    print("Image min     :", images.min().item())
    print("Image max     :", images.max().item())

    assert images.ndim == 4
    assert images.shape[1] == 3
    assert labels.ndim == 1
    assert images.shape[0] == labels.shape[0]

    assert torch.isfinite(images).all()
    assert torch.isfinite(labels.float()).all()

    print()
    print("DATASET TEST PASSED")


if __name__ == "__main__":
    main()