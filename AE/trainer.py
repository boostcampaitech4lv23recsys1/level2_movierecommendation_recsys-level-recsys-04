import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_dataloader,valid_dataloader,test_dataloader, args):
        self.args = args
        self.model = model.to(args.device)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

    
    def train(self):
        self.model.train()
        for i in range(self.args.epoch):
            train_loss = 0
            valid_loss = 0
            #train
            for batch in tqdm(self.train_dataloader):
                #breakpoint()
                #users = batch[0].to(self.args.device)
                #items = batch[1].to(self.args.device)
                # answers = batch[2].to(self.args.device)
                batch = batch.to(self.args.device)
                output = self.model(batch)
                cur_loss = self.loss(output, batch)
                
                self.optim.zero_grad()
                cur_loss.backward()
                self.optim.step()

                train_loss += cur_loss

            #valid
            for batch in tqdm(self.valid_dataloader):
                #breakpoint()
                #users = batch[0].to(self.args.device)
                #items = batch[1].to(self.args.device)
                # answers = batch[2].to(self.args.device)
                batch = batch.to(self.args.device)
                output = self.model(batch)
                cur_loss = self.loss(output, batch)
                
                self.optim.zero_grad()
                cur_loss.backward()
                self.optim.step()

                valid_loss += cur_loss
            print("Epoch = ",i ,"valid_loss", valid_loss, "train_loss",train_loss)
        return self.model


    def predict(self):
        self.model.eval()

        for batch in tqdm(self.test_dataloader):
            #breakpoint()
            #users = batch[0].to(self.args.device)
            #items = batch[1].to(self.args.device)
            # answers = batch[2].to(self.args.device)
            batch = batch.to(self.args.device)
            output = self.model(batch)
            
        
        # np.argpartition(output, -10)[-10:]
        #output.argsort()
        #output.argsort()[-10:]
        #breakpoint()
        return output.T.argsort()[:, -10:]