import sys
from pathlib import Path

from pathlib import Path
import sys

import torch

# Add the training folder itself to Python's import path
TRAINING_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING_DIR))

from face_dataset import create_dataloaders


def main():

    train_loader, val_loader, test_loader = create_dataloaders(
        root="datasets/face_manipulation",
        batch_size=2,
        num_workers=0,
    )

    print("Train batches :", len(train_loader))
    print("Val batches   :", len(val_loader))
    print("Test batches  :", len(test_loader))

    images, labels = next(iter(train_loader))

    print("Image shape   :", images.shape)
    print("Labels shape  :", labels.shape)
    print("Labels        :", labels.tolist())
    print("Image dtype   :", images.dtype)
    print("Image min     :", images.min().item())
    print("Image max     :", images.max().item())

    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)
    assert images.dtype == torch.float32

    print("\nFACE DATASET TEST PASSED")


if __name__ == "__main__":
    main()