import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 1. Accuracy comparison
# --------------------------------------------------

summary = pd.read_csv("results/summary.csv")

plt.figure(figsize=(8, 5))

plt.bar(
    summary["augmentation"],
    summary["mean_accuracy"],
    yerr=summary["std_accuracy"],
    capsize=5
)

plt.ylabel("Classification Accuracy (%)")
plt.xlabel("Augmentation")
plt.title("Mean CIFAR-10 Accuracy Across 3 Random Seeds")

plt.xticks(rotation=15)

plt.tight_layout()

plt.savefig(
    "results/accuracy_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 2. Training-loss comparison
# --------------------------------------------------

loss_files = glob.glob(
    "results/*_seed*_training_loss.csv"
)

all_losses = []

for file in loss_files:

    filename = os.path.basename(file)

    # Examples:
    # crop_seed42_training_loss.csv
    # gaussian_noise_seed123_training_loss.csv

    name = filename.replace(
        "_training_loss.csv",
        ""
    )

    augmentation, seed = name.rsplit(
        "_seed",
        1
    )

    data = pd.read_csv(file)

    data["augmentation"] = augmentation
    data["seed"] = int(seed)

    all_losses.append(data)


loss_data = pd.concat(
    all_losses,
    ignore_index=True
)


mean_losses = (
    loss_data
    .groupby(
        ["augmentation", "epoch"]
    )["loss"]
    .mean()
    .reset_index()
)


plt.figure(figsize=(8, 5))

for augmentation in (
    mean_losses["augmentation"].unique()
):

    subset = mean_losses[
        mean_losses["augmentation"]
        == augmentation
    ]

    plt.plot(
        subset["epoch"],
        subset["loss"],
        marker="o",
        label=augmentation
    )


plt.xlabel("Pretraining Epoch")
plt.ylabel("Mean Contrastive Loss")
plt.title(
    "Contrastive Training Loss Across Augmentations"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    "results/training_loss_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 3. Combine seed-42 t-SNE figures
# --------------------------------------------------

augmentations = [
    "crop",
    "rotation",
    "color_jitter",
    "gaussian_noise"
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(12, 10)
)

axes = axes.flatten()

for axis, augmentation in zip(
    axes,
    augmentations
):

    image_file = (
        "results/"
        + augmentation
        + "_seed42_tsne.png"
    )

    image = plt.imread(
        image_file
    )

    axis.imshow(image)

    axis.set_title(
        augmentation.replace(
            "_",
            " "
        ).title()
    )

    axis.axis("off")


plt.suptitle(
    "t-SNE Representation Comparison — Seed 42"
)

plt.tight_layout()

plt.savefig(
    "results/tsne_comparison_seed42.png",
    dpi=300
)

plt.close()


print("Created:")
print("results/accuracy_comparison.png")
print("results/training_loss_comparison.png")
print("results/tsne_comparison_seed42.png")