import subprocess
import sys


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


for augmentation in augmentations:

    for seed in seeds:

        print()
        print("=" * 60)
        print("Augmentation:", augmentation)
        print("Seed:", seed)
        print("=" * 60)

        command = [
            sys.executable,
            "main.py",
            "--augmentation",
            augmentation,
            "--pretrain-epochs",
            "10",
            "--evaluation-epochs",
            "5",
            "--batch-size",
            "128",
            "--seed",
            str(seed)
        ]

        subprocess.run(
            command,
            check=True
        )