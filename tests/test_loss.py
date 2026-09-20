import torch

from src.loss import ContrastiveLoss


loss_function = ContrastiveLoss()

z1 = torch.randn(
    8,
    128
)

z2 = torch.randn(
    8,
    128
)

loss = loss_function(
    z1,
    z2
)

print("Loss:")
print(loss)

print()

print("Loss value:")
print(loss.item())