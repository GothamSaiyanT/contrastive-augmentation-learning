import torch

from src.model import SimCLRModel


model = SimCLRModel()

fake_images = torch.randn(
    8,
    3,
    32,
    32
)

features, projections = model(fake_images)


print("Input shape:")
print(fake_images.shape)

print()

print("Feature shape:")
print(features.shape)

print()

print("Projection shape:")
print(projections.shape)