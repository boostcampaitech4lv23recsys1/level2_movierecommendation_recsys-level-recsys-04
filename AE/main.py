from model import AutoEncoder 
from dataset import Dataset 
from trainer import Trainer 
import argparse
import torch
import pandas as pd
import json
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)

    # gpu
    # parser.add_argument("--cuda", default=True, type=bool)

    # dimensions
    parser.add_argument("--hidden_dim", default=200, type=int)
    parser.add_argument("--batch_size", default=16384, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epoch", default=10, type=int)
    
    #train vaild ratio
    parser.add_argument("--valid_ratio", default=0.1, type=float)


    # neg sample ratio
    #parser.add_argument("--neg_ratio", default=1, type=int)

    args = parser.parse_args()
    
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #device 3개다 퀴즈쇼.
    #모델, 학습, dataloader
    
    #dataloder

    #data = json.loads(open(args.data_dir + "Ml_item2attributes.json").readline())
    data = pd.read_csv(args.data_dir + "train_ratings.csv")
    dataset = Dataset(args,data)
    train_dataloader = DataLoader(dataset.train_matrix, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset.valid_matrix, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset.test_matrix, batch_size=args.batch_size, shuffle=True)
    
    #print(loader.__getitem__(1))
    #breakpoint()

    #model
    model = AutoEncoder(args,data)
    
    train = Trainer(model, train_dataloader,valid_dataloader,test_dataloader, args)
    #breakpoint()

    train.train()
    predict = train.predict()
    #breakpoint()
    #inference
    predict = pd.DataFrame(predict.cpu().numpy())
    predict = predict.reset_index()
    predict['index'] = dataset.user_encoder.inverse_transform(predict['index'])
    #breakpoint()
    for i in range(10):
        predict[i] = dataset.item_encoder.inverse_transform(predict[i])
    #predict['index'] = dataset.user_encoder.inverse_transform(predict['index'])
    
    #breakpoint()

    submision = []
    predict.to_csv("predict.csv", index=False)
    for i in range(len(predict)):
        user = predict['index'].iloc[i]
        for j in range(0,10):
            submision.append([user, predict[j].iloc[i]])
    

    ''''''
    submision = pd.DataFrame(submision)
    #breakpoint()
    submision.columns = ['user', 'item']
    submision.to_csv("../output/AE.csv", index=False)
    return print("END")


if __name__ == '__main__':
    main()
