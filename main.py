import argparse

from src.experiment import Experiment


def main():

    parser = argparse.ArgumentParser(
        description="Run one SimCLR experiment"
    )

    parser.add_argument(
        "augmentation",
        required=True,
        choices=[
            "none",
            "crop",
            "rotation",
            "color_jitter",
            "gaussian_noise"
        ]
    )

    parser.add_argument(
        "pretrain-epochs",
        type=int,
        default=10
    )

    parser.add_argument(
        "evaluation-epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "batch-size",
        type=int,
        default=128
    )

    args = parser.parse_args()

    experiment = Experiment(
        augmentation_name=args.augmentation,
        pretrain_epochs=args.pretrain_epochs,
        evaluation_epochs=args.evaluation_epochs,
        batch_size=args.batch_size
    )

    experiment.run()


if __name__ == "__main__":
    main()