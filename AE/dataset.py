import torch
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self, args, data) -> None:
    # data type =  dict
        self.args = args
        self.data = data
        self.valid_ratio = args.valid_ratio
        self.num_user = len(data['user'].unique())
        self.num_item = len(data['item'].unique())
        
        self.user_encoder, self.item_encoder = self.generate_encoder_decoder()
        
        self.train_data, self.valid_data = train_test_split(data, test_size = self.valid_ratio, random_state = 42)
        self.train_matrix = self.make_matrix(self.train_data)
        self.valid_matrix = self.make_matrix(self.valid_data)
        self.test_matrix = self.make_matrix(self.data)
        
        
    def generate_encoder_decoder(self):
        """
        encoder, decoder 생성

        Args:
            col (str): 생성할 columns 명
        Returns:
            dict: 생성된 user encoder, decoder
        """
        user_encoder = preprocessing.LabelEncoder()
        user_encoder.fit(self.data['user'])
        item_encoder = preprocessing.LabelEncoder()
        item_encoder.fit(self.data['item'])
        #item_encoder.transform() -> 0~ num ]
        #item_encoder.inverse_transform()
        return user_encoder, item_encoder    
        
    def make_matrix(self, data):
        # mat[user][item]
        matrix = np.zeros((self.num_item, self.num_user), dtype = np.float32)
        self.users = self.user_encoder.transform(self.data['user'])         
        self.items = self.item_encoder.transform(self.data['item'])
        # breakpoint()       
            
        for user, item in zip(self.users, self.items):
            matrix[item][user] = 1
       #matrix = torch.tensor(matrix, dtype = torch.float32)
        return matrix
    
        
    def __getitem__(self, index):
        cur_idx = self.data.iloc[index]
        cur_tensors = (
            torch.tensor(cur_idx[0], dtype=torch.long),
            torch.tensor(cur_idx[1], dtype=torch.long)
        )
        return cur_tensors

    def __len__(self):
        return len(self.data)