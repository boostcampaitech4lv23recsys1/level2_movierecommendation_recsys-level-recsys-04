from datetime import datetime
from pytz import timezone

import pandas as pd

import torch
from torch.utils.data import DataLoader

import argparse

from utils import (
    get_factorized_index,
    load_csv
)
from preprocess import get_user_seq
from datasets import MFDataset
from models import MatrixFactorization
from trainer import Trainer



def main():
    parser = argparse.ArgumentParser()

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)

    # gpu
    # parser.add_argument("--cuda", default=True, type=bool)

    # dimensions
    parser.add_argument("--emb_hid_dim", default=32, type=int)
    parser.add_argument("--batch_size", default=16384, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=1, type=int)

    # neg sample ratio
    parser.add_argument("--neg_ratio", default=1, type=int)

    args = parser.parse_args()
    
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    args.data_file = args.data_dir + "train_ratings.csv"
    
    df = load_csv(args.data_file)
    df = get_factorized_index(args, df)
    df = get_user_seq(args, df)

    dataset = MFDataset(args, df)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MatrixFactorization(args)
    trainer = Trainer(model, dataloader, args)
    trainer.train()
    df_top_k = trainer.test(args, df)
    
    ''' submission '''
    now = datetime.now().astimezone(timezone('Asia/Seoul'))
    now = now.strftime("%Y_%m_%d_%H_%M")
    df_top_k.to_csv('../submission/' + now + '.csv', index=False)
    ''''''
    return


if __name__ == '__main__':
    main()
