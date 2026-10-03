import random
import numpy as np
import pandas as pd
import torch

from src.data import CIFAR10Data
from src.model import SimCLRModel
from src.evaluator import LinearEvaluator


augmentations = [
    "crop",
    "rotation",
    "color_jitter",
    "gaussian_noise"
]

seeds = [
    42,
    123,
    999
]


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


for augmentation in augmentations:

    for seed in seeds:

        print()
        print("=" * 60)
        print("Augmentation:", augmentation)
        print("Seed:", seed)
        print("=" * 60)

        # Make linear evaluation reproducible
        set_seed(seed)

        data = CIFAR10Data(
            batch_size=128
        )

        train_loader, test_loader = (
            data.get_classification_loaders()
        )

        model = SimCLRModel()

        checkpoint_file = (
            "checkpoints/"
            + augmentation
            + "_seed"
            + str(seed)
            + "_model.pt"
        )

        model.load_state_dict(
            torch.load(
                checkpoint_file,
                map_location=device
            )
        )

        evaluator = LinearEvaluator(
            encoder=model.encoder,
            feature_size=model.feature_size,
            device=device
        )

        evaluator.train(
            train_loader=train_loader,
            epochs=5
        )

        accuracy = evaluator.test(
            test_loader
        )

        result_file = (
            "results/"
            + augmentation
            + "_seed"
            + str(seed)
            + "_results.csv"
        )

        result = pd.read_csv(
            result_file
        )

        result["accuracy"] = accuracy

        result.to_csv(
            result_file,
            index=False
        )

        print(
            "Corrected accuracy:",
            round(accuracy * 100, 2),
            "%"
        )