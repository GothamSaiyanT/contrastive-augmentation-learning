from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from .augmentations import AugmentationFactory

class CIFAR10Data:
    #the constructor
    def __init__(self,data_directory="./data",batch_size = 128):
        self.data_directory = data_directory
        self.batch_size = batch_size
        #create the augmentation factory
        self.augmentation_factory = AugmentationFactory()


    #create a contrastive loader
    #that prepares the data for SimCLR training
    def get_contrastive_loader(self,augmentation_name):
        transform = self.augmentation_factory.create(
            augmentation_name
        )
        #Load CiFAR-10
        dataset = CIFAR10(
            root =self.data_directory,
            train =True,
            download = True,
            transform = transform
        )
        #create batches
        data_loader = DataLoader(
            dataset,
            batch_size = self.batch_size,
            shuffle=True,
            drop_last=True
        )
        return data_loader

    def get_classification_loaders(self):

        transform = (
            self.augmentation_factory.create_evaluation_transform()

        )

        train_dataset = CIFAR10(
            root =self.data_directory,
            train = True,
            download = True,
            transform = transform
        )

        test_dataset = CIFAR10(
            root =self.data_directory,
            train = False,
            download = True,
            transform = transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader,test_loader