import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    #create a temp to control the similarity differences
    def __init__(self,temperature = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self,z1,z2):
        #find the batch size
        batch_size = z1.size(0)

        #join the projections from view1 and view2
        projections = torch.cat(
            [z1,z2],
            dim=0,
        )

        #normalize the projection vectors
        projections = F.normalize(
            projections,
            dim=1
        )

        #compare every projection with every other projection
        similarity_matrix = torch.matmul(
            projections,
            projections.T
        )

        #control how strong the similarity 
        similarity_matrix = (
            similarity_matrix/ self.temperature
        )

        #prevent an image comparing with itself
        # create a diagonal mask
        mask = torch.eye(
            2 * batch_size,
            dtype=torch.bool,
            device=z1.device
        )
        #ignore
        similarity_matrix = similarity_matrix.masked_fill(
            mask,
            float("-inf")
        )

        #making sure of the correct matching of views
     
        targets = torch.cat([
            torch.arange(
                batch_size,
                2 * batch_size,
                device=z1.device
            ),
            torch.arange(
                0,
                batch_size,
                device=z1.device
            )
        ])
        loss = F.cross_entropy(
            similarity_matrix,
            targets
        )

        return loss