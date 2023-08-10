library(quantreg)
library(kernlab)
library(truncnorm)
#install.packages('svmpath')
library(svmpath)

# simulation setting
set.seed(2022021328)
n <- 500; p <- 50

x <- matrix(runif(n * p, -1, 1), n, p)   # x: 500 x 50
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]; 
x5 <- x[,5]; x6 <- x[,6]; x7 <- x[,7]; x8 <- x[,8]

A <- sample(c(-1, 1), n, replace = TRUE, prob = c(0.5, 0.5))

t1 <- 0.442*(1-x1-x2)*A   # linear: 0.612
t2 <- (x2 - 0.25*x1^2-1)*A   # poly: 0.604
t3 <- (0.5 - x1^2 - x2^2)*(x1^2 + x2^2 - 0.3)*A   # sigmoid: 0.624
t4 <- (1 - x1^3 + exp(x3^2 + x5) +0.6 * x6 - (x7 + x8)^2)*A   # poly: 0.606



Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)


fit_ITR <- function(x, A, R, pi, kappa, kern, gam){
  n <- nrow(x)
  if (kern == 'linear') {
    eps <- 0.0001
    
    K <- poly.kernel(x, param.kernel = 1)
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H + eps*diag(n), a, b, l, u, r)
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u-1e-7)
    
    beta <- apply(alpha * A * x, 2, sum)  
    temp <- A[sv.index] - x[sv.index,] %*% beta
    beta0 <- mean(temp)
    
    sol <- list(kern = 'linear', alpha = alpha, coef = beta, intercept = beta0)
    return(sol)
  }
  else if (kern == 'poly') {
    eps <- 0.0001
    
    K <- poly.kernel(x, param.kernel = 2)   
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H + eps*diag(n), a, b, l, u, r)
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u-1e-7)   
    
    k <- K[, sv.index]  
    est_sv <- apply(alpha * A * k, 2, sum) 
    temp <- A[sv.index] - est_sv
    theta0 <- mean(temp)
    
    sol <- list(treat = A, train = x, kern = 'poly', alpha = alpha, intercept = theta0)
    return(sol)
  }
  else if (kern == 'radial') {
    eps <- 0.0001
    
    K <- radial.kernel(x)
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H + eps*diag(n), a, b, l, u, r)
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u-1e-7)
    
    k <- K[, sv.index]  
    est_sv <- apply(alpha * A * k, 2, sum)  
    temp <- A[sv.index] - est_sv
    theta0 <- mean(temp)
    
    sol <- list(treat = A, train = x, kern = 'radial', alpha = alpha, intercept = theta0)
    return(sol)
  }
  else if (kern == 'sigmoid') {
    eps <- 0.0001
    
    K <- tanh(gam * x %*% t(x))
    H <- K * (A %*% t(A))
    c <- rep(-1, n)
    a <- matrix(A, 1, n)
    b <- 0; r <- 0
    l <- rep(0, n); u <- kappa * R / pi
    obj <- ipop(c, H + eps*diag(n), a, b, l, u, r)
    
    alpha <- obj@primal
    sv.index <- which(l[1]+1e-7 < alpha & alpha < u-1e-7)
    theta <- apply(alpha * A * K, 2, sum) 
    
    k <- K[, sv.index]  
    est_sv <- apply(alpha * A * k, 2, sum)  
    temp <- A[sv.index] - est_sv
    theta0 <- mean(temp)
    
    sol <- list(gam = gam, treat = A, train = x, kern = 'sigmoid', alpha = alpha, intercept = theta0)
    return(sol)
  }
  
}

predict_ITR <- function(model, newdata){  
  if (model$kern == 'linear'){
    y.hat <- as.vector(ifelse(model$intercept + newdata %*% model$coef > 0, 1, -1))
  }
  else if (model$kern == 'poly'){
    K <- poly.kernel(model$train, newdata)
    est <- apply(model$alpha * model$treat * K, 2, sum) + model$intercept
    y.hat <- as.vector(ifelse(est > 0, 1, -1))
  }
  else if (model$kern == 'radial'){
    K <- radial.kernel(model$train, newdata)
    est <- apply(model$alpha * model$treat * K, 2, sum) + model$intercept
    y.hat <- as.vector(ifelse(est > 0, 1, -1))
  }
  else if (model$kern == 'sigmoid'){
    K <- tanh(model$gam * model$train %*% t(newdata))
    est <- apply(model$alpha * model$treat * K, 2, sum) + model$intercept
    y.hat <- as.vector(ifelse(est > 0, 1, -1))
  }
  return(y.hat)
}

set.seed(2022021328)
n <- 500
p <- 10; d <- 3   # rp

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0.5)
A <- sign(c(x %*% beta + e))

t1 <- 0.442*(1-x1-x2)*A 
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)


sim <- fit_ITR(x, A, R, 0.5, 1, kern = 'linear')
pred <- predict_ITR(sim, x)


# value function
val.func <- function(pred, R, A){  # A: treatment 추가
  pi.a <- sum(A == 1) / n
  value <- sum(R * (A == pred) / pi.a)
  return(value/n)
}

val.func(pred, R, A)
