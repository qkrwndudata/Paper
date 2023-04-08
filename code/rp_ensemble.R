## 정의한 ITR fitting 함수와 predict 함수에 RP Ensemble을 적용해 가장 많이 선택된 변수를 뽑아내는 알고리즘 구현 코드입니다.

# library 호출
library(Matrix)
library(plyr)

# define rp generate functions
generate_projection <- function(d, p){
  a <- matrix(0, p, d)
  sel <- sample(1:p, d, replace = F)
  for (i in 1:d){
    a[sel[i], i] <- 1
  }
  return(t(a))
}

# rp ensemble 적용
rp_ensemble <- function(x, A, R, B1, B2, d, kernf){   # alpha 나중에 추가
  
  n <- nrow(x)
  p <- ncol(x)

  max.value <- list()   # max.value: b1개의 가장 높은 value function을 갖는 RP 저장
  for (i in 1: B1){
    rps <- list()   # rps: B2개의 RP 저장
    values <- list()   # values: B2개의 value function 결과 저장
    for (j in 1: B2){
      ind <- sample(1:n, n, replace = T)
      x.new <- matrix(0, n, p)
      for (k in 1:n) x.new[k,] <- x[ind[k],]   # bootstrap으로 x.new 생성
      
      rp <- generate_projection(d, p)
      z <- x.new %*% t(rp)   
      
      classifier <- fit_ITR(z, A, R, 0.5, 0.05, kern = kernf)   
      pred <- predict_ITR(classifier, z)
      value <- val.func(pred, R)  
      
      rps[[j]] <- rp
      values[[j]] <- value
    }
    max.value[[i]] <- rps[[which.max(values)]]   # value function을 maximize하는 RP 선택
  }
  final <- as.matrix(ldply(max.value, rbind)) # B1개의 행렬을 rbind한 후 colsum으로 결과 확인해 가장 많이 선택된 10개 변수 선택
  vars <-order(colSums(final), decreasing = TRUE)[1:d]
  return(sort(vars))
}

rp_ensemble(x, A, R, 10, 10, 10, 'poly')
