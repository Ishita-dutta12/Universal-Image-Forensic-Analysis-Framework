from pathlib import Path
import csv

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IMD2020Dataset(Dataset):

    def __init__(self, manifest_path, train=False):

        self.manifest_path = Path(manifest_path)
        self.train = train

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {self.manifest_path}"
            )

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.samples = list(
                csv.DictReader(f)
            )

        if train:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        # Masks must NOT receive random augmentation independently.
        # They must remain spatially aligned with the image.
        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        row = self.samples[index]

        manipulated_path = Path(row["manipulated"])
        mask_path = Path(row["mask"])

        image = Image.open(
            manipulated_path
        ).convert("RGB")

        mask = Image.open(
            mask_path
        ).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        # Convert mask to binary.
        mask = (mask > 0.5).float()

        label = torch.tensor(
            1,
            dtype=torch.long
        )

        return image, mask, label


class IMD2020AuthenticDataset(Dataset):

    def __init__(self, manifest_path, train=False):

        self.manifest_path = Path(manifest_path)
        self.train = train

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        # One authentic image per unique folder.
        seen = set()
        self.samples = []

        for row in rows:

            folder = row["folder"]

            if folder in seen:
                continue

            seen.add(folder)

            self.samples.append(
                row["original"]
            )

        if train:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        image_path = Path(
            self.samples[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.image_transform(image)

        # Authentic image has no manipulated region.
        mask = torch.zeros(
            (1, 224, 224),
            dtype=torch.float32
        )

        label = torch.tensor(
            0,
            dtype=torch.long
        )

        return image, mask, label


class IMD2020CombinedDataset(Dataset):

    def __init__(self, manifest_path, train=False):

        self.manifest_path = Path(manifest_path)
        self.train = train

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as f:
            rows = list(
                csv.DictReader(f)
            )

        # Manipulated samples
        self.manipulated_samples = rows

        # One authentic sample per original folder
        seen = set()
        self.authentic_samples = []

        for row in rows:

            folder = row["folder"]

            if folder in seen:
                continue

            seen.add(folder)

            self.authentic_samples.append(
                row["original"]
            )

        if train:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])

        self.mask_transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=transforms.InterpolationMode.NEAREST
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):

        return (
            len(self.manipulated_samples)
            + len(self.authentic_samples)
        )

    def __getitem__(self, index):

        # ---------------------------------------------
        # Manipulated sample
        # ---------------------------------------------

        if index < len(self.manipulated_samples):

            row = self.manipulated_samples[index]

            image_path = Path(
                row["manipulated"]
            )

            mask_path = Path(
                row["mask"]
            )

            image = Image.open(
                image_path
            ).convert("RGB")

            mask = Image.open(
                mask_path
            ).convert("L")

            image = self.image_transform(image)

            mask = self.mask_transform(mask)

            mask = (mask > 0.5).float()

            label = torch.tensor(
                1,
                dtype=torch.long
            )

            return image, mask, label

        # ---------------------------------------------
        # Authentic sample
        # ---------------------------------------------

        authentic_index = (
            index
            - len(self.manipulated_samples)
        )

        image_path = Path(
            self.authentic_samples[authentic_index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        image = self.image_transform(image)

        mask = torch.zeros(
            (1, 224, 224),
            dtype=torch.float32
        )

        label = torch.tensor(
            0,
            dtype=torch.long
        )

        return image, mask, label


def create_imd_dataloaders(
    root="datasets/IMD2020",
    batch_size=2,
    num_workers=0,
):

    root = Path(root)

    train_dataset = IMD2020CombinedDataset(
        root / "train.csv",
        train=True
    )

    val_dataset = IMD2020CombinedDataset(
        root / "val.csv",
        train=False
    )

    test_dataset = IMD2020CombinedDataset(
        root / "test.csv",
        train=False
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
        test_loader
    )