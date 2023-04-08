# library
library(Matrix)
library(plyr)


# simulation setting
set.seed(2022021328)
n <- 500; p <- 50; d <- 10

x <- matrix(runif(n * p, -1, 1), n, p)   # x: 500 x 50
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]; x5 <- x[,5]
x6 <- x[,6]; x7 <- x[,7]; x8 <- x[,8]

A <- sample(c(-1, 1), n, replace = TRUE, prob = c(0.5, 0.5))

t1 <- 0.442*(1-x1-x2)*A   # linear: 0.612
t2 <- (x2 - 0.25*x1^2-1)*A   # poly: 0.604
t3 <- (0.5 - x1^2 - x2^2)*(x1^2 + x2^2 - 0.3)*A   # sigmoid: 0.624
t4 <- (1 - x1^3 + exp(x3^2 + x5) +0.6 * x6 - (x7 + x8)^2)*A   # poly: 0.598

Q <- 1 + 2*x1 + x2 + 0.5*x3 + t2
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)


# define rp generate functions
generate_projection <- function(d, p){
  a <- matrix(0, p, d)
  sel <- sample(1:p, d, replace = F)
  for (i in 1:d){
    a[sel[i], i] <- 1
  }
  return(t(a))
}




# rp ensemble
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
      value <- val.func(pred, R)   # value function이 모두 같게 나옴...: 해결해야
      
      rps[[j]] <- rp
      values[[j]] <- value
    }
    max.value[[i]] <- rps[[which.max(values)]]
  }
  final <- as.matrix(ldply(max.value, rbind)) # B1개의 행렬을 rbind한 후 colsum으로 결과 확인해 가장 많이 선택된 10개 변수 선택
  vars <-order(colSums(final), decreasing = TRUE)[1:d]
  return(sort(vars))
}

rp_ensemble(x, A, R, 10, 10, 10, 'poly')
