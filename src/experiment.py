import torch
import pandas as pd

from .data import CIFAR10Data
from .model import SimCLRModel
from .trainer import SimCLRTrainer
from .evaluator import LinearEvaluator
from .visualizer import TSNEVisualizer


class Experiment:

    def __init__(
        self,
        augmentation_name,
        pretrain_epochs=10,
        evaluation_epochs=5,
        batch_size=128
    ):
        self.augmentation_name = augmentation_name
        self.pretrain_epochs = pretrain_epochs
        self.evaluation_epochs = evaluation_epochs
        self.batch_size = batch_size

        self.device = self.get_device()

        self.data = CIFAR10Data(
            batch_size=self.batch_size
        )

        self.model = SimCLRModel()


    def get_device(self):

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")


    def run(self):

        print("=" * 50)
        print("Starting experiment")
        print("Augmentation:", self.augmentation_name)
        print("Device:", self.device)
        print("=" * 50)

        # STEP 1:
        # Prepare CIFAR-10 for contrastive training
        contrastive_loader = (
            self.data.get_contrastive_loader(
                self.augmentation_name
            )
        )

        # STEP 2:
        # Train SimCLR
        trainer = SimCLRTrainer(
            model=self.model,
            device=self.device
        )

        training_history = trainer.train(
            data_loader=contrastive_loader,
            epochs=self.pretrain_epochs
        )

        # STEP 3:
        # Prepare CIFAR-10 for classification
        train_loader, test_loader = (
            self.data.get_classification_loaders()
        )

        # STEP 4:
        # Evaluate the learned features
        evaluator = LinearEvaluator(
            encoder=self.model.encoder,
            feature_size=self.model.feature_size,
            device=self.device
        )

        evaluator.train(
            train_loader=train_loader,
            epochs=self.evaluation_epochs
        )

        accuracy = evaluator.test(
            test_loader
        )

        # STEP 5:
        # Create t-SNE visualization
        visualizer = TSNEVisualizer(
            encoder=self.model.encoder,
            device=self.device
        )

        visualizer.create_plot(
            data_loader=test_loader,
            output_file=(
                "results/"
                + self.augmentation_name
                + "_tsne.png"
            ),
            maximum_samples=1000
        )

        # STEP 6:
        # Save a simple summary
        results = pd.DataFrame([
            {
                "augmentation": self.augmentation_name,
                "accuracy": accuracy,
                "pretrain_epochs": self.pretrain_epochs,
                "evaluation_epochs": self.evaluation_epochs
            }
        ])

        results.to_csv(
            "results/"
            + self.augmentation_name
            + "_results.csv",
            index=False
        )

        print()
        print("Experiment finished.")
        print(
            "Final accuracy:",
            round(accuracy * 100, 2),
            "%"
        )

        return accuracy