import torch
import torch.nn as nn

import pandas as pd
class AutoEncoder(nn.Module):
    def __init__(self, args, data) -> None:
        super(AutoEncoder,self).__init__()
        self.args = args
        self.data = data
        self.hidden_dim = args.hidden_dim
        self.num_user = len(data['user'].unique())
        self.num_item = len(data['item'].unique())
        # self.num_item = set()
        # for item in data.items():
        #     self.num_item.add(item)
        # self.num_item = len(self.num_item)
        
        self.encoder = nn.Sequential(
            nn.Linear(self.num_user, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(self.hidden_dim, self.num_user)
        )
    def forward(self, x):
        #breakpoint()
        x = self.encoder(x)
        return self.decoder(x)

    