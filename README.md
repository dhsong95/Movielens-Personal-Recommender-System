# Movielens-Personal-Recommender-System
Movielens 개인화 추천 시스템 구축

# 1. Matrix Factorization Model

```
python matrix_factorization.py
```

- movielens 데이터 세트를 가지고 Matrix Factorization 기반 추천을 생성한다.

- Matrix Factorization 추천은 implicit의 ALS 추천 모델 라이브러리를 사용합니다.

- Matrix Factorization을 기반으로 이루어진 추천 결과를 npz 파일로 저장하여서 이후 Auto Encoder의 결과와 상세 비교합니다.

# 2. AutoEncdoer Model

```
python autoencoder_cf.py
```

- movielens 데이터 세트를 가지고 AutoEncoder 기반 추천을 생성한다.

- AutoEncoder는 Keras를 기반으로 만들었으며 신경망 구조는 `autoencoder.py`에 저장되어 있습니다.

- 입력에 대해서 20개의 Latent Feature로 압축한 이후 복원하는 방식을 사용합니다.

- 학습된 모델을 바탕으로 movielens 데이터를 사용해서 모델을 학습하고 추천 결과를 npz 파일로 저장합니다. 이는 Matrix Factorization 모델과 상세 비교하기 위해 사용합니다.


### 모델 비교
두 모델을 비교하기 위해서 N-Precision을 사용합니다. 이는 각 모델 당 N개의 추천 결과를 생성하고, 추천된 결과를 바탕으로 숨긴 영화를 얼마나 많이 맞히는 지 판단하는 것입니다.

분석 결과 Matrix Factorization은 100개중 10개, Auto Encoder는 100개중 5개를 맞히는 데 성공하였습니다. 수치상으로 비교하면 Auto Encoder가 현저히 떨어지는 성능을 보이는 것 같지만 세부 추천 결과를 살펴보면 꼭 그러한 것만은 아닌 것을 확인할 수 있습니다.

두 모델의 추천 결과를 세부 비교하기 위해서는 Interactive하게 실험할 수 있는 jupyter에서 실행하였습니다. 1번 사용자를 대상으로 한 추천 결과 확인하는 실험이다. [노트북 열기: Recommendation Results](./notebooks/Recommendation%20Results.ipynb)

---

# 3. Item2Vec Model

```
python item2vec_cf.py
```

- Item2Vec 모델을 학습하여서 영화에 대한 Embedding을 생성합니다.

- Item2Vec 모델은 `item2vec.py`에서 Keras를 사용해서 구현합니다.

- Model은 Embedding 차원을 100으로 하고, 중심 단어(영화)를 위한 Embedding과 맥락 단어(영화)를 위한 Embedding을 학습합니다. 최종 Embedding은 중심 단어(영화)를 위해 생성된 Embedding입니다.

- Embedding을 npz 파일로 저장하고 노트북에서 아이템 간의 유사도를 비롯한 실험을 진행합니다. [노트북 열기: Item Similarity](./notebooks/Item%20Similarity.ipynb)
