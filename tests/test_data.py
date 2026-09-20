from src.data import CIFAR10Data


data = CIFAR10Data(batch_size=8)

loader = data.get_contrastive_loader("crop")


for (view1, view2), labels in loader:

    print("View 1 shape:")
    print(view1.shape)

    print()

    print("View 2 shape:")
    print(view2.shape)

    print()

    print("Labels:")
    print(labels)

    break