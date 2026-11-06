import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)
import torch

from training_model import UIFAFTrainingModel


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

model = UIFAFTrainingModel(
    device=device,
    pretrained=True,
)

model.train()

# --------------------------------------------------
# Fake training image
# --------------------------------------------------

images = torch.randn(
    1,
    3,
    224,
    224,
    device=device,
)

# --------------------------------------------------
# Forward
# --------------------------------------------------

logits = model(images)

print("Logits shape:", logits.shape)
print("Logits:", logits)

# --------------------------------------------------
# Fake label
# 0 = Authentic
# --------------------------------------------------

label = torch.tensor(
    [0],
    dtype=torch.long,
    device=device,
)

# --------------------------------------------------
# Loss
# --------------------------------------------------

criterion = torch.nn.CrossEntropyLoss()

loss = criterion(
    logits,
    label,
)

print("Loss:", loss.item())

# --------------------------------------------------
# Backward
# --------------------------------------------------

loss.backward()

print("Backward pass: SUCCESS")

# --------------------------------------------------
# Check trainable gradients
# --------------------------------------------------

gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        if parameter.grad is not None:

            if torch.isfinite(parameter.grad).all():

                gradient_found = True

                print(
                    "Gradient OK:",
                    name,
                    parameter.grad.abs().mean().item(),
                )

                break

if not gradient_found:
    raise RuntimeError(
        "No valid gradients found in trainable modules."
    )

print("TRAINING FLOW TEST PASSED")