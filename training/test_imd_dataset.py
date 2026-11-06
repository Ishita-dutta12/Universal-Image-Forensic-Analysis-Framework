from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.imd_dataset import create_imd_dataloaders


def main():

    train_loader, val_loader, test_loader = (
        create_imd_dataloaders(
            root="datasets/IMD2020",
            batch_size=2,
            num_workers=0,
        )
    )

    print("======================================")
    print("IMD2020 DATASET TEST")
    print("======================================")

    print(
        "Train batches:",
        len(train_loader)
    )

    print(
        "Val batches:",
        len(val_loader)
    )

    print(
        "Test batches:",
        len(test_loader)
    )

    # Test validation loader because it uses
    # deterministic preprocessing.
    images, masks, labels = next(
        iter(val_loader)
    )

    print("\nImages:", images.shape)
    print("Masks :", masks.shape)
    print("Labels:", labels.tolist())

    print(
        "Image dtype:",
        images.dtype
    )

    print(
        "Mask dtype:",
        masks.dtype
    )

    print(
        "Mask min:",
        masks.min().item()
    )

    print(
        "Mask max:",
        masks.max().item()
    )

    print(
        "Mask positive pixels:",
        (masks > 0).sum().item()
    )

    assert images.shape == (
        2, 3, 224, 224
    )

    assert masks.shape == (
        2, 1, 224, 224
    )

    assert labels.shape == (2,)

    assert images.dtype == torch.float32

    assert masks.dtype == torch.float32

    assert set(
        labels.tolist()
    ).issubset({0, 1})

    print(
        "\nIMD2020 DATASET TEST PASSED"
    )


if __name__ == "__main__":
    main()