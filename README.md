# Movie Recommendation

![image](https://user-images.githubusercontent.com/68258495/211189168-0228f7ae-6a05-4691-a59b-e2a2e6cc8155.png)
# 👋 RECCAR 팀원 소개

| <img src="https://user-images.githubusercontent.com/79916736/207600031-b46e76d2-cba3-4c94-9fc3-d9f29cd3bef8.png" width=200> | <img src="https://user-images.githubusercontent.com/113089704/208005478-0501fcea-89e8-42cd-959a-226c3ddb5b63.jpg" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207601023-bbf9e64f-1447-41d8-991f-677593094592.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207600724-c140a102-39fc-4c03-8109-f214773a64fc.png" width=200> | <img src="https://user-images.githubusercontent.com/65999962/210237522-72198783-f40c-491b-b8a7-6e6badf6cc24.jpg" width=200> | <img src="https://user-images.githubusercontent.com/79916736/208005357-e98d106d-a207-4acd-ab4b-1abf7dbcb69f.png" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [배성재](https://github.com/SeongJaeBae)                                            |                                            [양승훈](https://github.com/Seunghoon-Schini-Yang)                                            |                                         [조수연](https://github.com/Suyeonnie)                                          |                                            [홍재형](https://github.com/secrett2633)                                            |                                            [황선태](https://github.com/HSUNEH)                                            |

<br />
<br />

# 🏆️ 프로젝트 목표
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/65529313/168472960-0eac76e2-4fe3-4ebc-b093-f9c0aab59859.png" /></p> -->
- 사용자의 영화 시청 이력 데이터를 바탕으로, 사용자가 다음에 시청할 영화 및 좋아할 영화를 추천

<br />
<br />

# 💻 활용 장비
- GPU Tesla V100-PCIE-32GB

<br />
<br />

# 🙋🏻‍♂️🏻‍♀️ 프로젝트 팀 구성 및 역할
- **김성연** : EDA / 전반적인 팀 프로젝트 타임라인 설정 / Rule-based 모델 설계 / 베이스라인 주석 추가 및 개선 작업
- **배성재** : AutoEncoder 및 glocal-K 모델 적용, EASE 모델 적용 및 파라미터 튜닝, 베이스라인 주석 추가
- **양승훈** : EDA / 베이스라인 주석 추가 및 개선 작업 / MF 모델 직접 구현 및 적용
- **조수연** : EASE 모델 적용 / lightFM 모델 적용 
- **홍재형** : AutoEncoder 적용 / 베이스라인 주석 추가 / 미션 제출 코드 작성
- **황선태** : 대회 미션 과제 주석 추가 / 베이스라인 주석 추가

<br />
<br />

# 👨‍👩‍👧‍👦 협업 방식
#### a. Zoom (Google Meet)
#### b. Notion
#### c. Slack
#### d. Google Docs

<br />
<br />

# 🎢 프로젝트 수행 절차

#### 1. EDA
#### 2. Rule-based Model
#### 3. SASRec Model (S3-Rec with no Pre-training)
#### 4. Hyperparameter Tuning (Wandb)
#### 5. Ensemble (SASRec + EASE)

<br />
<br />

# ⌨️ Model Architecture
```
자세한 진행 과정은 발표자료 내 PDF 파일을 참고해주세요!
```
## Rule-Base 모델
```
Model_ipynb/KSY_rulbase.ipynb 실행
```

## S3-Rec 모델 (Baseline)

영화 추천 대회를 위한 S3-Rec 베이스라인 코드입니다.<br>
다음 코드를 대회에 맞게 재구성 했습니다.

- 논문 링크: https://arxiv.org/abs/2008.07873
- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec

### Installation

```
pip install -r requirements.txt
```

### How to run
0. Encoding
   ```
   ensamble.ipynb로 인코딩 된 train_new 파일 생성
   ```
1. Pretraining
   ```
   python run_pretrain.py
   ```
2. Fine Tuning (Main Training)
   1. with pretrained weight
      ```
      python run_train.py --using_pretrain
      ```
   2. without pretrained weight
      ```
      python run_train.py
      ```
3. Inference
   ```
   python inference.py
   ```
4. Decoding
   ```
   ensamble.ipynb로 디코딩 된 submission.csv 파일 생성
   ```
   
<br />

## EASE
```
EASE/bae_EASE.ipynb 실행
```

## Ensemble
```
ensemble.ipynb 실행
```

<br />
<br />

# 💯 프로젝트 수행 결과 - 최종 Private 2등

- Private 리더보드 스코어 Recall@10 기준으로 순위 선정

|리더보드| Recall@10  |     순위     |
|:--------:|:------:|:----------:|
|public| 0.1869 |  **2위**   |
|private| 0.1699 | **최종 2위** |

![image](https://user-images.githubusercontent.com/68258495/211190290-e59bc060-2dc8-4660-a8ef-d03522c4c10b.png)






