import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import PretrainDataset
from models import S3RecModel
from trainers import PretrainTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json, 
    get_user_seqs_long,
    set_seed,
)


def main():
    parser = argparse.ArgumentParser() 

    # 데이터 경로와 네이밍 부분.
    parser.add_argument("--data_dir", default="../data/train/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Ml", type=str)

    # 모델 argument(하이퍼 파라미터)
    parser.add_argument("--model_name", default="Pretrain", type=str)

    parser.add_argument(
        "--hidden_size", type=int, default=256, help="hidden size of transformer model"
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=2, help="number of layers"
    )
    # hidden_size % num_attention_heads == 0 만족해야 함
    # modules.py -> SelfAttention 클래스 참고
    parser.add_argument("--num_attention_heads", default=2, type=int)
    # 활성화 함수. (default gelu => relu 변형)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu

    # dropout하는 prob 기준 값 정하기? (모델 본 사람이 채워줘.)
    parser.add_argument(
        "--attention_probs_dropout_prob",
        type=float,
        default=0.4,
        help="attention dropout p",
    )
    parser.add_argument(
        "--hidden_dropout_prob", type=float, default=0.3, help="hidden dropout p"
    )
    # models.py -> init_weights 함수
    # initialize model weight -> (mean=0, std=initializer_range) 로 초기화
    parser.add_argument("--initializer_range", type=float, default=0.02)
    # 최대 시퀀셜 길이 설정 (datasets.py)
    parser.add_argument("--max_seq_length", default=100, type=int)

    # train args, 트레이너 하이퍼파라미터
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    # pre train args, 프리트레이너 하이퍼파라미터
    parser.add_argument(
        "--pre_epochs", type=int, default=20, help="number of pre_train epochs"
    )
    parser.add_argument("--pre_batch_size", type=int, default=512)

    # sequence에서 item을 masking 처리할 확률 (=negative case로 처리할 확률) (datasets.py)
    # 이 값이 커지면, negative item 비율이 늘어남 (1일 경우 모두 negative)
    parser.add_argument("--mask_p", type=float, default=0.2, help="mask probability")
    parser.add_argument("--aap_weight", type=float, default=1, help="aap loss weight") # 0.2
    parser.add_argument("--mip_weight", type=float, default=0, help="mip loss weight") # 1.0
    parser.add_argument("--map_weight", type=float, default=0, help="map loss weight") # 1.0
    parser.add_argument("--sp_weight", type=float, default=0, help="sp loss weight") # 0.5

    # 옵티마이저 관련 하이퍼파라미터
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.8, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.9999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    # parser 형태로 argument를 입력받습니다.
    args = parser.parse_args()

    # 시드 고정 (utils.py 내 함수 존재)
    set_seed(args.seed)
    # output 폴더가 존재하는지 체크. 존재하지 않는다면 만들어줍니다. (utils.py 내 함수 존재)
    check_path(args.output_dir)

    # Pretrain 모델 기록용 파일 경로 저장합니다.
    args.checkpoint_path = os.path.join(args.output_dir, "Pretrain.pt")

    # GPU 관련 설정 해줍니다.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # 데이터 파일 불러오는 경로 설정합니다.
    args.data_file = args.data_dir + "train_new.csv" # "train_ratings.csv"
    item2attribute_file = args.data_dir + "item2attributes.json" # args.data_name + "_item2attributes.json"
    # user_seq : 2차원 아이템 id 리스트, max_item : 가장 높은 아이템 id, long_sequence : 1차원 아이템 id 리스트.
    # user_seq 예시 : [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
    # 자세한건 get_user_seqs_long 함수 내에 써놨습니다.
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)

    # item2attribute : dict(item_id : genre의 list), attribute_size : genre id의 가장 큰 값
    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)
    # item, genre id의 가장 큰 값 저장합니다.
    args.item_size = max_item + 2 # 6808(mask_item_id 까지 포함해서)
    args.mask_id = max_item + 1 # 6807
    args.attribute_size = attribute_size + 1  # 19

    args.item2attribute = item2attribute

    # S3RecModel 모델을 불러옵니다. (models.py 내 존재)
    model = S3RecModel(args=args)
    # pre트레이너 클래스를 불러옵니다. (trainers.py 내 존재)
    trainer = PretrainTrainer(model, None, None, None, args)

    # EarlyStopping 클래스를 불러옵니다. (utils.py 내 존재)
    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    # pre_epochs 만큼(defalut : 300) epochs를 수행합니다.
    for epoch in range(args.pre_epochs):
        # PretrainDataset 클래스를 불러옵니다. (datasets.py 내 존재)
        # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열 => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        # long_sequence : long_sequence : 1차원 아이템 id 리스트 => [1번 유저 item_id 리스트, 2번 유저 item_id 리스트]
        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        # RandomSampler : 데이터 셋을 랜덤하게 섞어줍니다. 인덱스를 반환해줍니다.
        pretrain_sampler = RandomSampler(pretrain_dataset)
        # 모델 학습을 하기 위한 데이터 로더를 만듭니다. 랜덤으로 섞고 배치 단위(defalut : 256)로 출력합니다.
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size
        )
        # trainer 내 pretrain 함수를 실행. 학습을 수행합니다.
        losses = trainer.pretrain(epoch, pretrain_dataloader)
        ## comparing `sp_loss_avg``
        # early_stopping 여부를 점검하여 에포크를 멈출지 살펴봅니다.
        early_stopping(np.array([-losses["aap_loss_avg"]]), trainer.model) # sp_loss_avg
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
