import torch
from itertools import islice

from src.data import CIFAR10Data
from src.model import SimCLRModel
from src.evaluator import LinearEvaluator


# Use CPU for this small test
device = torch.device("cpu")


# Load CIFAR-10
data = CIFAR10Data(
    batch_size=8
)

train_loader, test_loader = (
    data.get_classification_loaders()
)


# Use only a few batches for testing
small_train_loader = islice(
    train_loader,
    2
)

small_test_loader = islice(
    test_loader,
    2
)


# Create the SimCLR model
model = SimCLRModel()


# Create the evaluator
evaluator = LinearEvaluator(
    encoder=model.encoder,
    feature_size=model.feature_size,
    device=device
)


# Train only the simple classifier
evaluator.train(
    train_loader=small_train_loader,
    epochs=1
)


# Test it
accuracy = evaluator.test(
    small_test_loader
)


print()
print("Test accuracy:")
print(accuracy)