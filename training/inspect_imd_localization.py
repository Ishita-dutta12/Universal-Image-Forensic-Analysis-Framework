from pathlib import Path
import sys

import torch
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from training.training_model import UIFAFTrainingModel
from training.imd_dataset import IMD2020Dataset


MANIFEST = Path("datasets/IMD2020/test.csv")
OUTPUT_DIR = Path(
    "outputs/evaluation/imd_localization"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("======================================")
    print("IMD2020 LOCALIZATION INSPECTION")
    print("======================================")

    print("Device:", device)

    # ---------------------------------------------------------
    # Load one manipulated sample
    # ---------------------------------------------------------

    dataset = IMD2020Dataset(
        MANIFEST,
        train=False
    )

    image, mask, label = dataset[0]

    print("Image:", image.shape)
    print("Mask :", mask.shape)
    print("Label:", label.item())

    # ---------------------------------------------------------
    # Load model
    # ---------------------------------------------------------

    model = UIFAFTrainingModel(
        pretrained=True
    ).to(device)

    model.eval()

    # ---------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------

    image_batch = image.unsqueeze(0).to(device)

    with torch.no_grad():

        logits, localization_logits = model(
        image_batch,
        return_localization=True
    )

    print("Logits shape:", logits.shape)
    print("Logits:", logits)

    print(
        "Localization logits shape:",
        localization_logits.shape
        )
     

    print(
    "Localization logits min:",
    localization_logits.min().item()
    )

    print(
    "Localization logits max:",
    localization_logits.max().item()
    )

    # ---------------------------------------------------------
    # Convert image tensor for visualization
    # ---------------------------------------------------------

    image_np = (
        image.permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    mask_np = (
        mask.squeeze(0)
        .cpu()
        .numpy()
    )

    # ---------------------------------------------------------
    # Save input + ground truth mask
    # ---------------------------------------------------------

    plt.figure(
        figsize=(10, 5)
    )

    plt.subplot(1, 2, 1)

    plt.imshow(image_np)

    plt.title(
        "IMD2020 Manipulated Image"
    )

    plt.axis("off")

    plt.subplot(1, 2, 2)

    plt.imshow(
        mask_np,
        cmap="gray"
    )

    plt.title(
        "Ground Truth Manipulation Mask"
    )

    plt.axis("off")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR /
        "sample_ground_truth.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print(
        "\nGround-truth visualization saved:"
    )

    print(output_path)

    print("\n======================================")
    print("INSPECTION COMPLETE")
    print("======================================")


if __name__ == "__main__":
    main()