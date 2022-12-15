import torch
import torch.nn as nn


class Trainer:
    def __init__(self, model, dataloader, args):
        self.args = args
        self.model = model
        self.dataloader = dataloader
        
        self.loss = nn.BCEWithLogitsLoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    
    def train(self):
        for i in range(self.args.epoch):
            for user, item, answer in self.dataloader:
                pass
        return
