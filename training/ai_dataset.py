from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import transforms

class AIGenerationDataset(Dataset):

    def __init__(
        self,
        csv_path,
        image_size=224,
        train=False,
    ):

        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.csv_path}"
            )

        self.data = pd.read_csv(self.csv_path)

        required_columns = {
            "image_path",
            "label",
        }

        missing = required_columns - set(self.data.columns)

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

        # Training augmentation
        if train:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.RandomHorizontalFlip(
                    p=0.5
                ),

                transforms.RandomRotation(
                    degrees=5
                ),

                transforms.ToTensor(),
            ])

        # Validation / Test
        else:

            self.transform = transforms.Compose([
                transforms.Resize(
                    (image_size, image_size)
                ),

                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = Path(row["image_path"])
        label = int(row["label"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            raise RuntimeError(
                f"Could not load image: {image_path}"
            ) from e

        image = self.transform(image)

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label


def create_dataloaders(
    root="datasets/ai_generation",
    batch_size=2,
    num_workers=0,
):

    root = Path(root)

    # Training augmentation ON
    train_dataset = AIGenerationDataset(
        root / "train.csv",
        train=True,
    )

    # Validation augmentation OFF
    val_dataset = AIGenerationDataset(
        root / "val.csv",
        train=False,
    )

    # Test augmentation OFF
    test_dataset = AIGenerationDataset(
        root / "test.csv",
        train=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
    )