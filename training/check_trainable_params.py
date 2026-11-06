import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from training_model import UIFAFTrainingModel
from training_model import UIFAFTrainingModel


model = UIFAFTrainingModel(device="cuda")

frozen = 0
trainable = 0

print("\nTRAINABLE MODULES")
print("=" * 60)

for name, parameter in model.named_parameters():

    if parameter.requires_grad:
        trainable += parameter.numel()

        if name.startswith("backbone"):
            raise RuntimeError(
                f"ConvNeXt parameter is unexpectedly trainable: {name}"
            )

    else:
        frozen += parameter.numel()


print("Trainable parameters:", trainable)
print("Frozen parameters:", frozen)

print("\nChecking ConvNeXt...")

backbone_trainable = [
    name
    for name, parameter in model.named_parameters()
    if name.startswith("backbone")
    and parameter.requires_grad
]

if backbone_trainable:
    raise RuntimeError(
        "ConvNeXt is not completely frozen."
    )

print("ConvNeXt: FROZEN")

print("\nPHASE 1 PARAMETER CHECK PASSED")