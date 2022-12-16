import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataloader, args):
        self.args = args
        self.model = model.to(args.device)
        self.dataloader = dataloader
        
        self.loss = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    
    def train(self):
        self.model.train()
        for i in range(self.args.epoch):
            train_loss = 0

            for batch in tqdm(self.dataloader):
                users = batch[0].to(self.args.device)
                items = batch[1].to(self.args.device)
                answers = batch[2].to(self.args.device)
                output = self.model(users, items)
                cur_loss = self.loss(output, answers)
                
                self.optim.zero_grad()
                cur_loss.backward()
                self.optim.step()

                train_loss += cur_loss

            breakpoint()
        return
