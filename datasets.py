import random

import torch
from torch.utils.data import Dataset

from utils import neg_sample


class PretrainDataset(Dataset):
    def __init__(self, args, user_seq, long_sequence):
        """
        Args:
            args (class): 전체 파일 args 총 집합 클래스.
            user_seq (list): 2차원 아이템 id 리스트, [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
            long_sequence (list): 1차원 아이템 id 리스트, [1번 유저 item_id 리스트, 2번 유저 item_id 리스트, ..]
        """        
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length # 시퀀셜 최대 길이(default=50)
        self.part_sequence = []
        # 아래 split_sequence 함수 참고
        self.split_sequence()

    def split_sequence(self):
        """
        이 함수를 통해서,
        part_sequence에 저장되는 sequence의 길이는 모두 max_len보다 작거나 같아짐
        """
        # self.user_seq : [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        for seq in self.user_seq: # seq : 유저마다 item_id 리스트. 
            lens = (len(seq[:-1]) // self.max_len) + 1
            for i in range(lens):
                self.part_sequence.append(seq[-(i+1)*self.max_len - 1: -i * self.max_len - 1])

    def __len__(self):
        # part_sequence 길이 반환
        return len(self.part_sequence)

    def __getitem__(self, index):
        # sequence : part_sequence의 해당 index에 저장된 sequence
        sequence = self.part_sequence[index]  # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []  # mask_p보다 작은 경우, mask_id 값 들어감
        neg_items = []  # mask_p보다 작은 경우, 시청하지 않은 item 중에서 random하게 들어감
        # Masked Item Prediction
        # item_set : list -> set (중복 item 제거)
        item_set = set(sequence)
        # sequence 마지막 데이터는 제외 (이 for문 바로 뒤에서 마지막 데이터는 masking처리 해줌)
        for item in sequence[:-1]:
            # prob : 0 <= x < 1 사이 random float 값
            prob = random.random()
            if prob < self.args.mask_p:  # prob가 mask_p 보다 작을 경우 (negative case) (default=0.2)
                # item 번호 대신, mask_id 추가 (masking 처리)
                masked_item_sequence.append(self.args.mask_id) # mask_id : max_item(item_id 최댓값) + 1
                # item_set 안에 없는 item (negative sample) random하게 뽑아서 neg_items에 추가
                neg_items.append(neg_sample(item_set, self.args.item_size)) # neg_sample(utils.py)
            else:  # prob가 mask_p 보다 크거나 같을 경우
                masked_item_sequence.append(item) 
                neg_items.append(item)

        # add mask at the last position (맨 뒤 한 값은 무조건 마스킹.)
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:  # sequence 길이 2보다 작은 경우
            masked_segment_sequence = sequence

            pos_segment = sequence
            neg_segment = sequence
        else:  # sequence 길이 2 이상인 경우
            # 1 <= sample_length <= (sequence 길이 // 2)
            sample_length = random.randint(1, len(sequence) // 2)
            # pos & neg items 가져올 index 랜덤 뽑기
            # pos_segment & neg_segment 길이가 sample_length보다 작아지지 않게 하려는 작업인듯
            start_id = random.randint(0, len(sequence) - sample_length)
            # self.long_sequence : 1차원 아이템 id 리스트, [1번 유저 item_id 리스트, 2번 유저 item_id 리스트, ..]
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            # pos_segment 길이 = sample_length
            # 실제로 시청한 positive sample들 가져옴
            pos_segment = sequence[start_id : start_id + sample_length]
            # neg_segment 길이 = sample_length
            # long_sequence의 random 위치에서 연속된 sample_length 개수 가져옴
            # 이러면 실제로 시청한 item이 neg_segment에 들어갈 수 있지 않을까?
            neg_segment = self.long_sequence[
                neg_start_id : neg_start_id + sample_length
            ]
            # start_id ~ (start_id + sample_length) 인덱스 원소 값 = mask_id
            # 나머지는 sequence와 동일
            masked_segment_sequence = (
                sequence[:start_id]
                + [self.args.mask_id] * sample_length
                + sequence[start_id + sample_length :]
            )
            # start_id ~ (start_id + sample_length) 인덱스 원소 값 = sequence와 동일
            # 나머지는 mask_id
            pos_segment = (
                [self.args.mask_id] * start_id
                + pos_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )
            # pos_segment와 동일한 로직
            neg_segment = (
                [self.args.mask_id] * start_id
                + neg_segment
                + [self.args.mask_id] * (len(sequence) - (start_id + sample_length))
            )

        # length check (동일하지 않으면 assertion error 발생)
        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence (max_len 길이로 맞춰주기 위한 padding 작업)
        pad_len = self.max_len - len(sequence)
        # 리스트 앞쪽을 0으로 채워줌
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0] * pad_len + masked_segment_sequence
        pos_segment = [0] * pad_len + pos_segment
        neg_segment = [0] * pad_len + neg_segment
        
        # 길이 max_len 넘지 않게 해줌
        masked_item_sequence = masked_item_sequence[-self.max_len :]
        pos_items = pos_items[-self.max_len :]
        neg_items = neg_items[-self.max_len :]

        masked_segment_sequence = masked_segment_sequence[-self.max_len :]
        pos_segment = pos_segment[-self.max_len :]
        neg_segment = neg_segment[-self.max_len :]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        # 각 item 별 장르 정보 저장
        '''
        ex)
        pos_items = [2, 4]
        attribute_size = 3
        item2attribute = {2: [1, 3], 4: [2]}
        attributes = [[0, 1, 0, 1], [0, 0, 1, 0]]
        ''' 
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute: # 속성이 여러개인 경우도 고려.
                    attribute[a] = 1
            except: # 속성이 없을수도. (모든 아이템은 다 속성이 1개는 있으나 mask_id와 padding 값 0은 속성이 없음.)
                pass
            attributes.append(attribute)

        # length check
        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len

        # to tensor
        cur_tensors = (
            torch.tensor(attributes, dtype=torch.long),
            torch.tensor(masked_item_sequence, dtype=torch.long),
            torch.tensor(pos_items, dtype=torch.long),
            torch.tensor(neg_items, dtype=torch.long),
            torch.tensor(masked_segment_sequence, dtype=torch.long),
            torch.tensor(pos_segment, dtype=torch.long),
            torch.tensor(neg_segment, dtype=torch.long),
        )
        '''
        (길이), 예시, 설명 순.
        attributes : (max_len * attribute_size), [[0,0,1,0 ...], [0,0,0,1 ...] ...], pos_items 기준 속성 멀티핫인코딩.
        masked_item_sequence : (max_len), [0,0,47,119146, .. ,119146], 영화 id 기록(마스킹 중간에 되있음.)
        pos_items : (max_len), [0,0,47,58, .. ,2571], 영화 id 기록(앞 패딩 이외 전처리하지 않은 순수 데이터)
        neg_items : (max_len), [0,0,47,10, .. ,25], 영화 id 기록(masked_item_sequence 마스킹 부분이 네거티브 샘플로 대체.)

        아래 3개 sequence 변수는 일부 아이템 중간 부분을 또 샘플링함. (위 코드를 한번 봐야 암.)
        masked_segment_sequence : (max_len), [0,0,47,119146, .. ,119146], 패딩/실제/마스킹/실제
        pos_segment : (max_len), [0,0,47,58, .. ,2571], 패딩/마스킹/실제/마스킹
        neg_segment : (max_len), [0,0,47,10, .. ,25], 패딩/마스킹/네거티브/마스킹
        ''' 
        return cur_tensors

# train 데이터 셋(pretrain X)
class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):
        # index로 user_id 사용
        # user_id를 index로 사용
        user_id = index
        items = self.user_seq[index]

        # check data_type
        assert self.data_type in {"train", "valid", "submission"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4, 5]
        # target [1, 2, 3, 4, 5, 6]
        # answer [6]

        # submission [0, 1, 2, 3, 4, 5, 6]
        # answer None

        if self.data_type == "train":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]
        else:
            input_ids = items[:]
            target_pos = items[:]  # will not be used
            answer = []


        target_neg = []
        seq_set = set(items)
        # input_ids 길이만큼 target_neg에 negative samples 생성
        # 자세한건 neg_sample 함수 내에 써놨습니다.
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        # padding
        # max_len 길이에 맞춰서 앞쪽 0으로 채움
        # pad_len 값이 음수이면, [0] * pad_len = []
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # 길이 max_len으로 통일
        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        # check length
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        # to tensor
        if self.test_neg_items is not None:  # 현재 베이스라인 코드에서, test_neg_items가 None이 아닌 경우는 찾지 못했음
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long), # input_ids 대비 하나씩 밀림.
                torch.tensor(target_neg, dtype=torch.long), # input_ids 길이만큼 네거티브 샘플링.
                torch.tensor(answer, dtype=torch.long), # 마지막 값.
            )

        return cur_tensors

    def __len__(self):
        # user 수 반환
        return len(self.user_seq)



class SASRecTrainDataset(Dataset):
    def __init__(self, args, user_seq):
        # user_seq : 유저마다 따로 아이템 리스트 저장. 2차원 배열, => [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        self.args = args
        self.user_seq = user_seq
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.part_user = []
        # 아래 split_sequence 함수 참고
        self.split_sequence()

    def split_sequence(self):
        """
        이 함수를 통해서,
        part_sequence에 저장되는 sequence의 길이는 모두 max_len보다 작거나 같아짐
        """
        # self.user_seq : [[1번 유저 item_id 리스트], [2번 유저 item_id 리스트] .. ]
        # self.max_len '+ 1'을 사용하는 이유 : target_pos 위해 원래 표본 크기보다 하나 더 뽑아야함.

        # 최근 데이터를 많이 반영하는 코드
        
        ## test
        k = self.args.num_k
        a = 2
        for seq in self.user_seq: # seq : 유저마다 item_id 리스트.
            for i in range(k):
                self.part_sequence.append(seq[-(self.max_len+1) - (i*a): - (i*a)])
                self.part_user.append(seq)

            lens = ((len(seq) - (k*a)) // (self.max_len+1)) + 1
            for i in range(lens):
                self.part_sequence.append(seq[-(i+1)*(self.max_len+1) - (k*a): -i * (self.max_len+1) - (k*a)])
                self.part_user.append(seq)

        ## valid
        # k = self.args.num_k
        # a = 2
        # for seq in self.user_seq: # seq : 유저마다 item_id 리스트.
        #     for i in range(k):
        #         self.part_sequence.append(seq[-(self.max_len+1) - (1+i*a): - (1+i*a)])
        #         self.part_user.append(seq)

        #     lens = ((len(seq) - (1+k*a)) // (self.max_len+1)) + 1
        #     for i in range(lens):
        #         self.part_sequence.append(seq[-(i+1)*(self.max_len+1) - (1+k*a): -i * (self.max_len+1) - (1+k*a)])
        #         self.part_user.append(seq)

        # 일반적인 데이터 argument
        # for seq in self.user_seq: # seq : 유저마다 item_id 리스트. 
        #     lens = ((len(seq) - 1) // (self.max_len+1)) + 1
        #     for i in range(lens):
        #         self.part_sequence.append(seq[-(i+1)*(self.max_len+1) - 1: -i * (self.max_len+1) - 1])
        #         self.part_user.append(seq)

    def __getitem__(self, index):
        # sequence : part_sequence의 해당 index에 저장된 sequence
        sequence = self.part_sequence[index]  # pos_items
        user_item_list = self.part_user[index]
        input_ids = sequence[:-1]
        target_pos = sequence[1:]
        target_neg = []
        user_set = set(user_item_list)#[:-1]) # valid에 있는 데이터는 네거티브에 안걸림.
        # input_ids 길이만큼 target_neg에 negative samples 생성
        for _ in input_ids:
            target_neg.append(neg_sample(user_set, self.args.item_size))

        # padding
        # max_len 길이에 맞춰서 앞쪽 0으로 채움
        # pad_len 값이 음수이면, [0] * pad_len = []
        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        # 길이 max_len으로 통일
        input_ids = input_ids[-self.max_len :]
        target_pos = target_pos[-self.max_len :]
        target_neg = target_neg[-self.max_len :]

        # check length
        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len
        
        user_id = 0 # 아무값이나 넣음. 어짜피 안써서
        answer = 0 # 아무값이나 넣음. 어짜피 안써서
        # to tensor
        cur_tensors = (
            torch.tensor(user_id, dtype=torch.long),  # user_id for testing
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long), # input_ids 대비 하나씩 밀림.
            torch.tensor(target_neg, dtype=torch.long), # input_ids 길이만큼 네거티브 샘플링.
            torch.tensor([1] * self.max_len, dtype=torch.long), # 마지막 값.
        )

        return cur_tensors

    def __len__(self):
        # user 수 반환
        return len(self.part_sequence)