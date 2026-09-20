import torch
from torch.optim import AdamW
from tqdm import tqdm

from .loss import ContrastiveLoss


class SimCLRTrainer:
    def __init__(self,model,device,learning_rate = 0.0003,temperature=0.5):
        self.model=model
        self.device = device

        #move the model to a selected device
        self.model.to(self.device)

        #optimizer updates the model weights
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = learning_rate
        )
        #contrastive loss
        self.loss_function = ContrastiveLoss(
            temperature=temperature
        )

    def train(self,data_loader,epochs):

        history = []

        for epoch in range(1,epochs + 1):

            self.model.train()

            total_loss=0
            total_images = 0

            progress_bar = tqdm(
                data_loader,
                desc = f"Epoch {epoch}/{epochs}"
            )

        for(view1,view2),_ in progress_bar:

            #move images to the same device as the model
            view1 = view1.to(self.device)
            view2 = view2.to(self.device)

            #clear old gradients
            self.optimizer.zero_grad()

            #send both views through the model
            _,z1 = self.model(view1)
            _,z2 = self.model(view2)

            #calculate contrastive loss
            loss = self.loss_function(
                z1,
                z2
            )

            #calculate gradients
            loss.backward()

            #update model weights
            self.optimizer.step()

            batch_size = view1.size(0)

            total_loss +=(
                loss.item() * batch_size
            )

            total_images += batch_size

            progress_bar.set_postfix(
                loss =f"{loss.item():.4f}"
            )

        average_loss = (
            total_loss/total_images
        )

        history.append({
            "epoch":epoch,
            "loss":average_loss
        })

        print(
            f"Epoch {epoch}:"
            f"average loss = {average_loss:.4f}"
        )

        return history