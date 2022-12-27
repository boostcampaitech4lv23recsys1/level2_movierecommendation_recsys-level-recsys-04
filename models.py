import torch
import torch.nn as nn

from modules import Encoder, LayerNorm


class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        # item embedding
        self.item_embeddings = nn.Embedding(
            args.item_size, args.hidden_size, padding_idx=0 # hidden_size : 64(defalut)
        )
        # attribute embedding ("genre")
        self.attribute_embeddings = nn.Embedding(
            args.attribute_size, args.hidden_size, padding_idx=0
        )
        # positional embedding
        # label 개수 <= max_seq_length
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # 트랜스포머의 일부 구성요소 시용한 인코더(사실은 디코더와 유사하다고 함.)
        # modules에 빡세게 구현되어있음.
        self.item_encoder = Encoder(args)
        # layer normalization
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # dropout
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.aap_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.mip_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.map_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.sp_norm = nn.Linear(args.hidden_size, args.hidden_size)
        self.criterion = nn.BCELoss(reduction="none")
        # initialize layers
        self.apply(self.init_weights)

    # AAP
    def associated_attribute_prediction(self, sequence_output, attribute_embedding):
        """
        :param sequence_output: [B L H]
        :param attribute_embedding: [arribute_num H]
        :return: scores [B*L tag_num] (각 영화마다 모든 장르 예측 값 배출)
        """
        sequence_output = self.aap_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num]

    # MIP sample neg items
    def masked_item_prediction(self, sequence_output, target_item):
        """
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        """
        # mip_norm : Linear hidden => hidden
        sequence_output = self.mip_norm(
            sequence_output.view([-1, self.args.hidden_size])
        )  # [B*L H]
        target_item = target_item.view([-1, self.args.hidden_size])  # [B*L H]
        score = torch.mul(sequence_output, target_item)  # [B*L H]
        return torch.sum(score, -1)  # [B*L], torch.sigmoid(torch.sum(score, -1)) 

    # MAP
    def masked_attribute_prediction(self, sequence_output, attribute_embedding):
        # map_norm : Linear hidden => hidden
        sequence_output = self.map_norm(sequence_output)  # [B L H]
        sequence_output = sequence_output.view(
            [-1, self.args.hidden_size, 1]
        )  # [B*L H 1]
        # [tag_num H] [B*L H 1] -> [B*L tag_num 1]
        # 실제 장르와 예측한 장르가 얼마나 일치하는지?
        score = torch.matmul(attribute_embedding, sequence_output)
        return torch.sigmoid(score.squeeze(-1))  # [B*L tag_num], torch.sigmoid(score.squeeze(-1))

    # SP sample neg segment
    def segment_prediction(self, context, segment):
        """
        :param context: [B H]
        :param segment: [B H]
        :return:
        """
        # sp_norm : Linear hidden => hidden
        context = self.sp_norm(context)  # [B H]
        score = torch.mul(context, segment)  # [B H]
        return torch.sum(score, dim=-1)  # [B], torch.sigmoid(torch.sum(score, dim=-1))

    
    def add_position_embedding(self, sequence):
        """_summary_
        입력 값에서 아이템 임베딩을 해준 뒤 포지션 임베딩을 더해줍니다.
        Args:
            sequence (tenser): (batch * max_len), 영화 id 기록
        Returns:
            sequence_emb (tenser): (batch * max_len * hidden_size)
        """        
        seq_length = sequence.size(1)  # max_len
        '''
        tensor([ 0, 1, ..., seq_length-1 ], device='cuda:0')
        '''
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=sequence.device
        )
        '''
        [B * L]
        tensor([[ 0, 1, ..., seq_length-1 ],
        [...] * (batch_szie-2),
        [ 0, 1, ..., seq_length-1 ]], device='cuda:0')
        '''
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # 아이템 임베딩 먼저하고 (B * L => B * L * H)
        item_embeddings = self.item_embeddings(sequence)
        # 포지션 임베딩 진행. (B * L => B * L * H)
        position_embeddings = self.position_embeddings(position_ids)
        # 아이템 임베딩 + 포지션 임베딩
        sequence_emb = item_embeddings + position_embeddings
        # Layer Nomralization [B * L * H]
        sequence_emb = self.LayerNorm(sequence_emb)
        # Dropout (결과 tensor 값들 일부 0으로 바뀜)
        sequence_emb = self.dropout(sequence_emb)  # [B * L * H]

        return sequence_emb

    def pretrain(
        self,
        attributes,
        masked_item_sequence,
        pos_items,
        neg_items,
        masked_segment_sequence,
        pos_segment,
        neg_segment,
    ):
        """
        Args:
            attributes (tensor): (batch * max_len * attribute_size), 아이템별 속성 멀티핫인코딩
            masked_item_sequence (tensor): (batch * max_len), 영화 id 기록(마스킹 중간에 되있음) 
            pos_items (tensor): (batch * max_len), 영화 id 기록(패딩 외 전처리하지 않은 순수 데이터)
            neg_items (tensor): (batch * max_len), 영화 id 기록(마스킹 부분이 네거티브 샘플로 대체)
            -------------------------------------------------------------------------
            (4번째 로스만을 위해 있는 3개의 변수.)
            masked_segment_sequence (tensor): (batch * max_len), 패딩/실제/마스킹/실제
            pos_segment (tensor): (batch * max_len), 패딩/마스킹/실제/마스킹 
            neg_segment (tensor): (batch * max_len), 패딩/마스킹/네거티브/마스킹
        Returns:
            aap_loss, mip_loss, map_loss, sp_loss: loss들
        """    
        # Encode masked sequence

        sequence_emb = self.add_position_embedding(masked_item_sequence)  # [B, L, H]
        # 패딩(0)값에 엄청난 마이너스 값(-1e8) 넣습니다.
        sequence_mask = (masked_item_sequence == 0).float() * -1e8  # [B, L]
        # sequence_mask : [B * L] => [B * 1 * L] => [B * 1 * 1 * L]
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1) 

        encoded_layers = self.item_encoder(
            sequence_emb, sequence_mask, output_all_encoded_layers=True
        )
        # [B L H], sequence_output : encoder 거친 최종 아웃풋.
        sequence_output = encoded_layers[-1]

        attribute_embeddings = self.attribute_embeddings.weight  # [attribute_size, H]

        # AAP
        aap_score = self.associated_attribute_prediction(
            sequence_output, attribute_embeddings
        )
        # aap_score : [B * L, attribute_size]
        # attributes : [B, L, attribute_size] => [B * L, attribute_size]
        aap_loss = self.criterion(
            aap_score, attributes.view(-1, self.args.attribute_size).float()
        )  # [B * L, attribute_size]
        # only compute loss at non-masked position(마스킹이거나 패딩인 아이템은 계산 제외)
        aap_mask = (masked_item_sequence != self.args.mask_id).float() * (
            masked_item_sequence != 0
        ).float()  # [B, L]
        # [B * L, attribute_size] * [B * L, 1]
        aap_loss = torch.sum(aap_loss * aap_mask.flatten().unsqueeze(-1))  # tensor(124563.9688, device='cuda:0', grad_fn=<SumBackward0>)

        # MIP
        pos_item_embs = self.item_embeddings(pos_items)
        neg_item_embs = self.item_embeddings(neg_items)
        # sequence_output : 트랜스포머를 통해 주변 영화와의 상호작용을 고려한 값.  [B, L, H]
        # pos_item_embs, neg_item_embs : 순수 그 영화의 임베딩.  [B, L, H]
        # 만약 네거티브 샘플링 된 영화라면 output이 원래 영화 임베딩과 멀어질 것을 기대.
        pos_score = self.masked_item_prediction(sequence_output, pos_item_embs)  # [B*L]
        neg_score = self.masked_item_prediction(sequence_output, neg_item_embs)  # [B*L]
        mip_distance = torch.sigmoid(pos_score - neg_score)  # [B*L]
        mip_loss = self.criterion(
            mip_distance, torch.ones_like(mip_distance, dtype=torch.float32)
        )  # [B*L]
        # 마스킹 된 부분만 로스를 구하겠다.
        mip_mask = (masked_item_sequence == self.args.mask_id).float()  # [B, L]
        mip_loss = torch.sum(mip_loss * mip_mask.flatten())  # tensor(2056.0708, device='cuda:0', grad_fn=<SumBackward0>)

        # MAP
        map_score = self.masked_attribute_prediction(
            sequence_output, attribute_embeddings  # [B L H], [attribute_size H]
        )
        # map_score : [B * L, attribute_size]
        # attributes : [B, L, attribute_size] => [B * L, attribute_size]
        map_loss = self.criterion(
            map_score, attributes.view(-1, self.args.attribute_size).float()
        )  # [B*L attribute_size]
        # 마스킹 된 영화의 장르를 잘 맞추는지 평가합니다.
        map_mask = (masked_item_sequence == self.args.mask_id).float()
        map_loss = torch.sum(map_loss * map_mask.flatten().unsqueeze(-1))

        # SP
        # segment context
        segment_context = self.add_position_embedding(masked_segment_sequence)  # [B L H]
        segment_mask = (masked_segment_sequence == 0).float() * -1e8 # 패딩(0) 처리 위해서
        # sequence_mask : [B * L] => [B * 1 * 1 * L]
        segment_mask = torch.unsqueeze(torch.unsqueeze(segment_mask, 1), 1)
        segment_encoded_layers = self.item_encoder(
            segment_context, segment_mask, output_all_encoded_layers=True
        )

        # take the last position hidden as the context
        segment_context = segment_encoded_layers[-1][:, -1, :]  # [B H]

        # pos_segment
        pos_segment_emb = self.add_position_embedding(pos_segment)
        pos_segment_mask = (pos_segment == 0).float() * -1e8
        pos_segment_mask = torch.unsqueeze(torch.unsqueeze(pos_segment_mask, 1), 1)
        pos_segment_encoded_layers = self.item_encoder(
            pos_segment_emb, pos_segment_mask, output_all_encoded_layers=True
        )
        pos_segment_emb = pos_segment_encoded_layers[-1][:, -1, :] # [B H]

        # neg_segment
        neg_segment_emb = self.add_position_embedding(neg_segment)
        neg_segment_mask = (neg_segment == 0).float() * -1e8
        neg_segment_mask = torch.unsqueeze(torch.unsqueeze(neg_segment_mask, 1), 1)
        neg_segment_encoded_layers = self.item_encoder(
            neg_segment_emb, neg_segment_mask, output_all_encoded_layers=True
        )
        neg_segment_emb = neg_segment_encoded_layers[-1][:, -1, :]  # [B H]

        pos_segment_score = self.segment_prediction(segment_context, pos_segment_emb)
        neg_segment_score = self.segment_prediction(segment_context, neg_segment_emb)

        sp_distance = torch.sigmoid(pos_segment_score - neg_segment_score)

        sp_loss = torch.sum(
            self.criterion(
                sp_distance, torch.ones_like(sp_distance, dtype=torch.float32)
            )
        )  # tensor(359.4898, device='cuda:0', grad_fn=<SumBackward0>)

        return aap_loss, mip_loss, map_loss, sp_loss

    # Fine tune
    # same as SASRec
    def finetune(self, input_ids):
        # attention_mask : [B, L], 패딩된 값은 0 / 아닌 값은 1인 마스킹 행렬 만들기.
        attention_mask = (input_ids > 0).long()
        # extended_attention_mask : [B, 1, 1, L]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(
            2
        )  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        # subsequent_mask : 상 삼각행렬. [[[0, 1, 1, .. 1], [0, 0, 1, .. 1], ... , [0,0, ... 0]]], [1, L, L]
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        # subsequent_mask : 하 삼각행렬. [[[[1, 0, 0, .. 0], [1, 1, 0, .. 0], ... , [1,1, ... 1]]]], [1, 1, L, L]
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        # [B, 1, 1, L] * [1, 1, L, L] => [B, 1, L, L]
        # extended_attention_mask : 패딩아닌 것만 1
        # subsequent_mask : 하나의 시퀀셜 영화기록 L을 L * L로 확장. 이전 기록 마스킹 하는 식으로 확장.
        # 두 마스킹을 곱하면 패딩과 이전 기록 마스킹을 동시에 하는 마스킹 텐서 탄생.
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        # 마스킹 된 값 -10000 곱하기. 마스킹 안된 값은 0.
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb, extended_attention_mask, output_all_encoded_layers=True
        )
        sequence_output = item_encoded_layers[-1]
        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            # Linear & Embedding weight mean=0, std=initializer_range로 초기화
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            # LayerNorm weight=1, bias=0 으로 초기화
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            # Bias 존재하는 Linear bias=0 으로 초기화
            module.bias.data.zero_()
