Random Projection Ensemble 기법을 사용해 ITR 분류 모형의 변수 선택을 진행하는 연구 과정을 기록하는 repository 입니다. 

## 개념 설명

- ### Random Projection Ensemble
> 고차원 데이터를 저차원으로 축소하는 데 사용되는 비지도 학습 기법 중 하나로, 데이터의 고차원 공간에서 임의의 방향 벡터들을 선택하고, 이 벡터들을 이용하여 데이터를 저차원 공간으로 투영하는 과정을 여러 번 반복하되 각각 다른 임의의 벡터들을 사용하여 데이터를 투영합니다. 이렇게 생성된 다양한 저차원 데이터들을 합쳐서 Ensemble을 형성하는 방법입니다.

- ### ITR(Individualized Treatment Rule)
> 개인의 특성, 질병의 특성, 그리고 다양한 치료 옵션 등에 대한 정보를 고려하여 개인마다 가장 적합한 치료 전략을 찾는 통계 방법론입니다.




## 연구 목표

1. 고차원 데이터를 학습하는 ITR 분류 모형을 구현하되, 선형(linear)/비선형(polynomial, rbf, sigmoid) 학습이 모두 가능하도록 설정
2. 해당 모형에 RP Ensemble을 적용하여 선택된 변수와 true parameter 비교해 일치 여부 확인
3. LASSO 등 기존 변수 선택 모형과 성능 비교


#### 참고자료
- Estimating Individualized Treatment Rules Using Outcome Weighted Learning(Yingqi Zhao, Donglin Zeng, A. John Rush, Michael R. Kosorok, Journal of the American Statistical Association, Vol. 107, No. 499, pp. 1106-1118)
- On Sparse representation for Optimal Individualized Treatment Selection with Penalized Outcome Weighted Learning(Rui Song, Michael Kosorok, Donglin Zeng, Yingqi Zhao, Eric Laber, Ming Yuan, NIH, 2015;4(1):59-68)
- Random-projection ensemble classification(Timothy I. Cannings, Richard J. Samworth, Journal of the Royal Statistical Society. Series B (Statistical Methodology), Vol. 79, No. 4 (2017), pp. 959-1035)
