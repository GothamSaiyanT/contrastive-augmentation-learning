import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


class TSNEVisualizer:

    def __init__(
        self,
        encoder,
        device
    ):
        self.encoder = encoder
        self.device = device

        self.encoder.to(self.device)


    def collect_features(
        self,
        data_loader,
        maximum_samples=1000
    ):

        self.encoder.eval()

        all_features = []
        all_labels = []

        total_samples = 0

        with torch.no_grad():

            for images, labels in data_loader:

                images = images.to(self.device)

                features = self.encoder(images)

                all_features.append(
                    features.cpu().numpy()
                )

                all_labels.append(
                    labels.numpy()
                )

                total_samples += labels.size(0)

                if total_samples >= maximum_samples:
                    break

        features = np.concatenate(
            all_features,
            axis=0
        )

        labels = np.concatenate(
            all_labels,
            axis=0
        )

        features = features[
            :maximum_samples
        ]

        labels = labels[
            :maximum_samples
        ]

        return features, labels


    def create_plot(
        self,
        data_loader,
        output_file,
        maximum_samples=1000
    ):

        features, labels = self.collect_features(
            data_loader,
            maximum_samples
        )

        # First reduce from 512 dimensions
        # to 50 dimensions
        pca = PCA(
            n_components=50
        )

        reduced_features = pca.fit_transform(
            features
        )

        # Then reduce from 50 dimensions
        # to 2 dimensions
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            random_state=42
        )

        points = tsne.fit_transform(
            reduced_features
        )

        # Create the graph
        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            points[:, 0],
            points[:, 1],
            c=labels,
            s=10
        )

        plt.title(
            "t-SNE of Learned Representations"
        )

        plt.xlabel(
            "Dimension 1"
        )

        plt.ylabel(
            "Dimension 2"
        )

        plt.tight_layout()

        output_directory = os.path.dirname(output_file)

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        plt.savefig(
            output_file
        )

        plt.close()

        print(
            "t-SNE plot saved to:",
            output_file
        )