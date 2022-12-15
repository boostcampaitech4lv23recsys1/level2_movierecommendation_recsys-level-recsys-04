import torch
from torch.utils.data import Dataset, DataLoader
import random


class MFDataset(Dataset):
    def __init__(self, args, data):
        self.args = args
        self.user_seq = data.groupby('user')['item'].apply(list)
        self.item_cnt = self.args.item_cnt
        self.neg_ratio = args.neg_ratio

    def __getitem__(self, index):
        user_id = index
        item_list = self.user_seq[index]
        item_len = len(item_list)
        item_set = set(item_list)

        neg_len = int(item_len * self.neg_ratio)
        for _ in range(neg_len):
            cur_item = random.randint(0, self.item_cnt-1)
            while cur_item in item_set:
                cur_item = random.randint(0, self.item_cnt-1)
            item_list.append(cur_item)
            item_set.add(cur_item)
        
        answer = [1] * item_len + [0] * neg_len
        user_list = [user_id] * (item_len + neg_len)

        cur_tensors = (
            torch.tensor(user_list, dtype=torch.long),
            torch.tensor(item_list, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long)
        )
        return cur_tensors


    def __len__(self):
        return self.args.user_cnt


