from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FaceManipulationDataset(Dataset):

    def __init__(self, root, split="train", train=False):

        self.root = Path(root) / split
        self.train = train

        self.samples = []

        # Real = 0
        # Fake = 1

        for image_path in sorted((self.root / "real").glob("*.jpg")):
            self.samples.append((image_path, 0))

        for image_path in sorted((self.root / "fake").glob("*.jpg")):
            self.samples.append((image_path, 1))

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

        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def create_dataloaders(
    root="datasets/face_manipulation",
    batch_size=2,
    num_workers=0,
):

    train_dataset = FaceManipulationDataset(
        root=root,
        split="train",
        train=True,
    )

    val_dataset = FaceManipulationDataset(
        root=root,
        split="val",
        train=False,
    )

    test_dataset = FaceManipulationDataset(
        root=root,
        split="test",
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

    return train_loader, val_loader, test_loader