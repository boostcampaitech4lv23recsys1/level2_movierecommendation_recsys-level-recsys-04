import numpy as np
import pandas as pd
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

        return self.model


    def test(self, args, df):
        self.model.eval()

        df_pos = df.loc[df['answer'] == 1]

        user_map = args.user_map
        item_map = args.item_map

        df_grpby = df_pos.groupby('user')['item'].apply(list)

        user_embedding = self.model.user_embedding.weight
        item_embedding = self.model.item_embedding.weight

        users = []
        items = []

        for user, user_emb in tqdm(enumerate(user_embedding)):
            user_probs = torch.matmul(user_emb, item_embedding.T)
            user_probs = user_probs.detach().cpu().numpy()
            user_probs[df_grpby[user]] = -np.inf

            top_k = np.argpartition(user_probs, -10)[-10:]

            user_real = int(user_map[user])
            item_real = item_map[top_k]

            users.extend([user_real] * 10)
            items.extend(list(item_real))

        df_top_k = pd.DataFrame({'user': users, 'item': items}, dtype=int)
        return df_top_k
