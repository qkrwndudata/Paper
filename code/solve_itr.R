## ipop()을 사용해 ITR classifier를 fitting 및 predict하는 함수 구현 코드 입니다.
## ipop()은 QP 알고리즘을 사용해 제약 조건 하에서 최적화 문제를 풀기 위해 사용되는 코드로 대표적으로 SVM 등의 모형이 해당 함수를 사용합니다.


# 라이브러리 호출
library(quantreg)
library(kernlab)
library(truncnorm)
library(svmpath)

# (linear/nonlinear)ITR fitting function 
fit_ITR <- function(x, A, R, pi, kappa, kern, gamma){
  n <- nrow(x)
  # linear 적합: 아래 K부터 l, u까지의 값은 제약식(lagrange multipliers)을 dual problem으로 풀어 정의했습니다.
  if (kern == 'linear') {
    K <- poly.kernel(x, param.kernel = 1)
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H, a, b, l, u, r)   # ipop solve
    
    alpha <- obj@primal   # dual solutions
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u[1]-1e-7)   # support vectors
    
    beta <- apply(alpha * A * x, 2, sum)   # primal solution
    temp <- A[sv.index] - x[sv.index,] %*% beta  
    beta0 <- mean(temp)   # intercept
    
    sol <- list(kern = 'linear', alpha = alpha, coef = beta, intercept = beta0)
    return(sol)
  }
  # polynomial 적합
  else if (kern == 'poly') {
    K <- poly.kernel(x, param.kernel = 2)   
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H, a, b, l, u, r)   
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u[1]-1e-7)   
    
    k <- K[, sv.index]  
    est_sv <- apply(alpha * A * k, 2, sum) 
    temp <- A[sv.index] - est_sv
    theta0 <- mean(temp)
    
    sol <- list(treat = A, train = x, kern = 'poly', alpha = alpha, intercept = theta0)
    return(sol)
  }
  # rbf 적합
  else if (kern == 'radial') {
    K <- radial.kernel(x)
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H, a, b, l, u, r)
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u[1]-1e-7)
    
    k <- K[, sv.index]  
    est_sv <- apply(alpha * A * k, 2, sum)  
    temp <- A[sv.index] - est_sv
    theta0 <- mean(temp)
    
    sol <- list(treat = A, train = x, kern = 'radial', alpha = alpha, intercept = theta0)
    return(sol)
  }
  # sigmoid 적합
  else if (kern == 'sigmoid') {
    K <- tanh(gamma * x %*% t(x))
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H, a, b, l, u, r)
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u[1]-1e-7)
    theta <- apply(alpha * A * K, 2, sum) 
    
    k <- K[, sv.index]  
    est_sv <- apply(alpha * A * k, 2, sum)  
    temp <- A[sv.index] - est_sv
    theta0 <- mean(temp)
    
    sol <- list(gamma = gamma, treat = A, train = x, kern = 'sigmoid', alpha = alpha, intercept = theta0)
    return(sol)
  }
  
}

# fitting한 모델로 예측하는 함수
predict_ITR <- function(model, newdata){  
  if (model$kern == 'linear'){
    y.hat <- as.vector(ifelse(model$intercept + newdata %*% model$coef > 0, 1, -1))
  }
  else if (model$kern == 'poly'){
    K <- poly.kernel(newdata, model$train)
    est <- apply(model$alpha * model$treat * K, 2, sum) + model$intercept
    y.hat <- as.vector(ifelse(est > 0, 1, -1))
  }
  else if (model$kern == 'radial'){
    K <- radial.kernel(newdata, model$train)
    est <- apply(model$alpha * model$treat * K, 2, sum) + model$intercept
    y.hat <- as.vector(ifelse(est > 0, 1, -1))
  }
  else if (model$kern == 'sigmoid'){
    K <- tanh(model$gamma * newdata %*% t(model$train))
    est <- apply(model$alpha * model$treat * K, 2, sum) + model$intercept
    y.hat <- as.vector(ifelse(est > 0, 1, -1))
  }
  return(y.hat)
}

sim <- fit_ITR(x, A, R, 0.5, 0.05, kern = 'poly', gamma = 0.1)
pred <- predict_ITR(sim, x)
