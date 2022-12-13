import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset
from models import S3RecModel
from trainers import FinetuneTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser()

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # 모델 argument(하이퍼 파라미터)
    parser.add_argument("--model_name", default="Finetune_full", type=str)
    parser.add_argument(
        "--hidden_size", type=int, default=64, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    parser.add_argument("--num_attention_heads", default=2, type=int)
    # 활성화 함수. (default gelu => relu 변형)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu

    # dropout하는 prob 기준 값 정하기? (모델 본 사람이 채워줘.)
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.5,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p"
    )
    # 모델 파라미터 initializer 범위 설정? (모델 본 사람이 채워줘.)
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # 최대 시퀀셜 길이 설정
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args, 트레이너 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument(
        "--batch_size", type=int, default=256, help="number of batch_size"
    )
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # 옵티마이저 관련 하이퍼파라미터
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # 프리트레인 된 모델을 사용할 것인지 체크.
    parser.add_argument("--using_pretrain", action="store_true")

    # parser 형태로 argument를 입력받습니다.
    args = parser.parse_args()

    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)
    # output 폴더가 존재하는지 체크. 존재하지 않는다면 만들어줍니다. (utils.py 내 함수 존재)
    check_path(args.output_dir)

    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # 데이터 파일 불러오는 경로 설정합니다.
    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    
    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    # max_item : 가장 큰 item_id, matrix 3개 : 유저-아이템 희소행렬
    # valid : 유저마다 마지막 2개 영화 시청기록 뺌, test : 유저마다 마지막 1개 영화 시청기록 뺌,
    # 자세한건 get_user_seqs 함수 내에 써놨습니다.
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    # item2attribute : dict(item_id : genre의 list), attribute_size : genre id의 가장 큰 값
    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    # item, genre id의 가장 큰 값 저장합니다.
    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args, (model_name : Finetune_full, data_name : Ml, output_dir : output/)
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # item2attribute : dict(item_id : genre의 list)
    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    # valid_rating_matrix : 유저마다 마지막 2개 영화 시청기록을 뺀 희소행렬.
    args.train_matrix = valid_rating_matrix

    # 모델 기록용 파일 경로 저장합니다.
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    # SASRecDataset 클래스를 불러옵니다. (datasets.py 내 존재)
    # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    # RandomSampler : 데이터 셋을 랜덤하게 섞어줍니다. 인덱스를 반환해줍니다.
    train_sampler = RandomSampler(train_dataset)
    # 모델 학습을 하기 위한 데이터 로더를 만듭니다. 랜덤으로 섞고 배치 단위(defalut : 256)로 출력합니다.
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    # eval(valid)도 마찬가지 입니다. data_type만 valid로 설정되있습니다.
    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    # test도 마찬가지 입니다. data_type만 test로 설정되있습니다.
    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    # S3RecModel 모델을 불러옵니다. (models.py 내 존재)
    model = S3RecModel(args=args)

    # Finetune 트레이너 클래스를 불러옵니다. (trainers.py 내 존재)
    # 트레이너에 모델, train, eval, test 데이터 로더 넣어줍니다. 
    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    # 프리트레인 모델 사용했는지 여부 출력.
    print(args.using_pretrain)
    if args.using_pretrain: # 프리트레인을 했는가?
        # 프리트레인 기록 불러오기.
        pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    # EarlyStopping 클래스를 불러옵니다. (utils.py 내 존재)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)


if __name__ == "__main__":
    main()
