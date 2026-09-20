import torch.nn as nn

#we will imoirt a ccn for images
from torchvision.models import resnet18

class SimCLRModel(nn.Module):
    def __init__(self,projection_size = 128):
        super().__init__()

        #create the encoder
        encoder = resnet18(weights=None)

        #since we do not want the final classifier instead we want feactues whci are small like 32*32 Cifar then we will mordify it
        encoder.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        #remove max pooling since it reduces the image dimensions but cifar is already reduced
        encoder.maxpool = nn.Identity()

        #find feacture size
        self.feature_size = encoder.fc.in_features
        encoder.fc = nn.Identity()

        self.encoder = encoder

        #create a projection head
        self.projection_head = nn.Sequential(
            nn.Linear(
                self.feature_size,
                self.feature_size
            ),
            nn.ReLU(),
            nn.Linear(
                self.feature_size,
                projection_size
            )
        )

    def forward(self,image):
        features = self.encoder(image)
        projection = self.projection_head(features)
        return features,projection
    #so here we have f->useful learned r->later for eva
    #and projection ->use during contrastive->used by loss of cont
