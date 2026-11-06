import csv
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CelebHqFaceDataset(Dataset):
    def __init__(
        self,
        root="datasets/CelebHQ-FM",
        split="train",
        train=False,
        seed=42,
    ):
        self.root = Path(root)
        self.train = train
        manifest_path = self.root / "Train.csv"

        with manifest_path.open(
            newline="",
            encoding="utf-8-sig",
        ) as manifest:
            reader = csv.DictReader(manifest)
            required_columns = {"PATH", "IDENTITY", "EDIT_NAME"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{manifest_path} must contain columns "
                    f"{sorted(required_columns)}"
                )
            rows = list(reader)

        identities = sorted({row["IDENTITY"] for row in rows})
        random.Random(seed).shuffle(identities)

        train_end = int(len(identities) * 0.8)
        val_end = int(len(identities) * 0.9)
        identity_splits = {
            "train": set(identities[:train_end]),
            "val": set(identities[train_end:val_end]),
            "test": set(identities[val_end:]),
        }
        if split not in identity_splits:
            raise ValueError(
                f"Unknown split {split!r}; expected train, val, or test"
            )

        self.samples = []
        for row in rows:
            if row["IDENTITY"] not in identity_splits[split]:
                continue

            image_path = self.root / row["PATH"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image listed in {manifest_path} was not found: "
                    f"{image_path}"
                )

            label = 0 if row["EDIT_NAME"] == "none" else 1
            self.samples.append(
                (image_path, label, row["IDENTITY"])
            )

        if not self.samples:
            raise ValueError(
                f"No CelebHQ-FM samples found for split {split!r}"
            )

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label, _ = self.samples[index]
        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    root="datasets/CelebHQ-FM",
    batch_size=2,
    num_workers=0,
    seed=42,
):
    train_dataset = CelebHqFaceDataset(
        root=root,
        split="train",
        train=True,
        seed=seed,
    )
    val_dataset = CelebHqFaceDataset(
        root=root,
        split="val",
        seed=seed,
    )
    test_dataset = CelebHqFaceDataset(
        root=root,
        split="test",
        seed=seed,
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

    return train_loader, val_loader, test_loader
