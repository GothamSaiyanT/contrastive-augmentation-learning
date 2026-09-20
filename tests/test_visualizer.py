import torch

from src.data import CIFAR10Data
from src.model import SimCLRModel
from src.visualizer import TSNEVisualizer


device = torch.device("cpu")


data = CIFAR10Data(
    batch_size=64
)

_, test_loader = (
    data.get_classification_loaders()
)


model = SimCLRModel()


visualizer = TSNEVisualizer(
    encoder=model.encoder,
    device=device
)


visualizer.create_plot(
    data_loader=test_loader,
    output_file="results/test_tsne.png",
    maximum_samples=500
)