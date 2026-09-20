import torch
from itertools import islice

from src.data import CIFAR10Data
from src.model import SimCLRModel
from src.trainer import SimCLRTrainer


# Use CPU for now
device = torch.device("cpu")


# Load CIFAR-10
data = CIFAR10Data(
    batch_size=8
)

loader = data.get_contrastive_loader(
    "crop"
)


# Only use ONE batch for this test
small_loader = islice(
    loader,
    1
)


# Create the model
model = SimCLRModel()


# Create the trainer
trainer = SimCLRTrainer(
    model=model,
    device=device
)


# Train for one epoch,
# but only using our one batch
history = trainer.train(
    data_loader=small_loader,
    epochs=1
)


print()
print("Training history:")
print(history)