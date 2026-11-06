import csv

from PIL import Image

from training.celeb_dataset import CelebHqFaceDataset, create_dataloaders


def _write_fixture(root):
    root.mkdir()
    rows = []
    for identity in range(10):
        for edit_name in ("none", "smile"):
            relative_path = f"train/{identity}/{identity}_{edit_name}.jpg"
            image_path = root / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(image_path)
            rows.append({
                "PATH": relative_path,
                "IDENTITY": str(identity),
                "EDIT_NAME": edit_name,
            })

    with (root / "Train.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as manifest:
        writer = csv.DictWriter(
            manifest,
            fieldnames=("PATH", "IDENTITY", "EDIT_NAME"),
        )
        writer.writeheader()
        writer.writerows(rows)


def test_celeb_dataset_splits_by_identity_and_maps_labels(tmp_path):
    root = tmp_path / "CelebHQ-FM"
    _write_fixture(root)

    datasets = {
        split: CelebHqFaceDataset(root, split=split)
        for split in ("train", "val", "test")
    }

    identities = {
        split: {sample[2] for sample in dataset.samples}
        for split, dataset in datasets.items()
    }
    assert identities["train"].isdisjoint(identities["val"])
    assert identities["train"].isdisjoint(identities["test"])
    assert identities["val"].isdisjoint(identities["test"])

    train_samples = datasets["train"].samples
    labels_by_edit = {
        path.stem.rsplit("_", 1)[-1]: label
        for path, label, _ in train_samples
    }
    assert labels_by_edit == {"none": 0, "smile": 1}

    image, label = datasets["train"][0]
    assert image.shape == (3, 224, 224)
    assert label.item() in (0, 1)


def test_celeb_create_dataloaders_return_expected_shapes(tmp_path):
    root = tmp_path / "CelebHQ-FM"
    _write_fixture(root)

    train_loader, val_loader, test_loader = create_dataloaders(
        root=root,
        batch_size=2,
    )
    images, labels = next(iter(train_loader))

    assert images.shape == (2, 3, 224, 224)
    assert labels.shape == (2,)
    assert len(val_loader.dataset) > 0
    assert len(test_loader.dataset) > 0
