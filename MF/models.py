import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    def __init__(self, args) -> None:
        super(MatrixFactorization, self).__init__()
        self.user_cnt = args.user_cnt
        self.item_cnt = args.item_cnt
        
        self.emb_hid_dim = args.emb_hid_dim

        self.user_embedding = nn.Embedding(self.user_cnt, self.emb_hid_dim)
        self.item_embedding = nn.Embedding(self.item_cnt, self.emb_hid_dim)

    
    def forward(self, user_id, item_id):
        user_embedding = self.user_embedding(user_id)
        item_embedding = self.item_embedding(item_id)

        output = torch.mul(user_embedding, item_embedding)
        output = torch.sum(output, axis=1)
        return output


