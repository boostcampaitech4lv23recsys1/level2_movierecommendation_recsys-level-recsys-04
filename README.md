# Movie Recommendation Baseline Code

![image](https://user-images.githubusercontent.com/68258495/211189168-0228f7ae-6a05-4691-a59b-e2a2e6cc8155.png)
## 팀원 소개

| <img src="https://user-images.githubusercontent.com/79916736/207600031-b46e76d2-cba3-4c94-9fc3-d9f29cd3bef8.png" width=200> | <img src="https://user-images.githubusercontent.com/113089704/208005478-0501fcea-89e8-42cd-959a-226c3ddb5b63.jpg" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207601023-bbf9e64f-1447-41d8-991f-677593094592.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207600724-c140a102-39fc-4c03-8109-f214773a64fc.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/208005357-e98d106d-a207-4acd-ab4b-1abf7dbcb69f.png" width=200> | <img src="https://user-images.githubusercontent.com/65999962/210237522-72198783-f40c-491b-b8a7-6e6badf6cc24.jpg" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [배성재](https://github.com/SeongJaeBae)                                            |                                            [양승훈](https://github.com/Seunghoon-Schini-Yang)                                            |                                         [조수연](https://github.com/Suyeonnie)                                          |                                            [황선태](https://github.com/HSUNEH)                                            |                                            [홍재형](https://github.com/secrett2633)                                            |

## Baseline

영화 추천 대회를 위한 S3-Rec 베이스라인 코드입니다.
다음 코드를 대회에 맞게 재구성 했습니다.

- 논문 링크: https://arxiv.org/abs/2008.07873
- 코드 출처: https://github.com/aHuiWang/CIKM2020-S3Rec

## Installation

```
pip install -r requirements.txt
```

## How to run

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
