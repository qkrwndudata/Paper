# library
library(Matrix)
library(plyr)
#install.packages('RPEnsemble')
library(RPEnsemble)
library(glmnet)
library(mvtnorm)


# simulation setting 1
set.seed(2022021328)
n <- 500; p <- 50; d <- 10

x <- matrix(runif(n * p, -1, 1), n, p)   # x: 500 x 50
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]; x5 <- x[,5]
x6 <- x[,6]; x7 <- x[,7]; x8 <- x[,8]; x9 <- x[,9]; x10 <- x[,10]

A <- sample(c(-1, 1), n, replace = TRUE, prob = c(0.5, 0.5))

t1 <- 0.442*(1-x1-x2)*A   # linear: 0.612
t2 <- (x2 - 0.25*x1^2-1)*A   # poly: 0.604
t3 <- (0.5 - x1^2 - x2^2)*(x1^2 + x2^2 - 0.3)*A   # sigmoid: 0.624
t4 <- (1 - x1^3 + exp(x3^2 + x5) +0.6 * x6 - (x7 + x8)^2)*A   # poly: 0.598

Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)


# simulations setting 2
n <- 40; p <- 50; d <- 5
x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]; x5 <- x[,5]
x6 <- x[,6]; x7 <- x[,7]; x8 <- x[,8]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
A <- sign(c(x %*% beta + e))

t1 <- 0.442*(1-x1-x2)*A 
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

x_train <- x[c(1:320),]
x_test <- x[c(321:400),]
A_train <- A[c(1:320)]
A_test <- A[c(321:400)]
R_train <- R[c(1:320)]
R_test <- R[c(321:400)]



# simulations setting 3
n <- 300; p <- 10; d <- 3
x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3];

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
A <- sign(c(x %*% beta + e))

t1 <- 0.442*(1-x1-x2)*A 
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

# simulations setting 4: simple simulation
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
  beta <- c(rep(1, d), rep(0, p-d))

  max.value <- list()   # max.value: b1개의 가장 높은 value function을 갖는 RP 저장
  for (i in 1: B1){
    rps <- list()   # rps: B2개의 RP 저장
    values <- list()   # values: B2개의 value function 결과 저장
    for (j in 1: B2){
      #ind <- sample(1:n, n, replace = T)
      #x.new <- matrix(0, n, p)
      #for (k in 1:n) x.new[k,] <- x[ind[k],]   # bootstrap으로 x.new 생성 -> 매번 새로 하는게 맞나?
      ### bootstrap하면 성능 안나옴 ###
      
      rp <- generate_projection(d, p)
      z <- x %*% t(rp)   
      
      classifier <- fit_ITR(z, A, R, 0.5, 0.1, kern = kernf, gam = 0.5)   # pi.a   
      pred <- predict_ITR(classifier, z)   # 새로운 dataset으로 예측하는 함수도 짜보기
      value <- val.func(pred, R, A)  
      
      rps[[j]] <- rp
      values[[j]] <- value
    }
    max.value[[i]] <- rps[[which.max(values)]]
  }
  #final <- as.matrix(ldply(max.value, rbind)) # B1개의 행렬을 rbind한 후 colsum으로 결과 확인해 가장 많이 선택된 10개 변수 선택
  #vars <-order(colSums(final), decreasing = TRUE)[1:d]
  final <- rep(0, p)
  for (i in 1:B1){
    final <- final + apply(max.value[[i]], 2, sum)
  }
  result <- list(output = final/B1)
  #result <- list(output = final/B1, rps <- max.value)
  return(result)
}

start_time <- Sys.time()
(obj <- rp_ensemble(x, A, R, 50, 20, 3, 'linear'))
end_time <- Sys.time()
end_time - start_time
(obj <- rp_ensemble(x_train, A_train, R_train, 50, 20, 3, 'linear'))


rp_ensemble.2 <- function(x, A, R, B1, B2, d){   # alpha 나중에 추가
  
  n <- nrow(x)
  p <- ncol(x)
  beta <- c(rep(1, d), rep(0, p-d))
  
  max.value <- list()   # max.value: b1개의 가장 높은 value function을 갖는 RP 저장
  for (i in 1: B1){
    rps <- list()   # rps: B2개의 RP 저장
    values <- list()   # values: B2개의 value function 결과 저장
    for (j in 1: B2){
      #ind <- sample(1:n, n, replace = T)
      #x.new <- matrix(0, n, p)
      #for (k in 1:n) x.new[k,] <- x[ind[k],]   # bootstrap으로 x.new 생성 -> 매번 새로 하는게 맞나?
      ### bootstrap하면 성능 안나옴 ###
      
      rp <- generate_projection(d, p)
      z <- x %*% t(rp)   
      
      classifier <- fit_ITR_GD.2(z, A, R, 0.5, 1, learn_rate = 0.001, max_iter = 1000)   # pi.a   
      pred <- predict_ITR.2(classifier, z)   # 새로운 dataset으로 예측하는 함수도 짜보기
      value <- val.func(pred, R, A)  
      
      rps[[j]] <- rp
      values[[j]] <- value
    }
    max.value[[i]] <- rps[[which.max(values)]]
  }
  #final <- as.matrix(ldply(max.value, rbind)) # B1개의 행렬을 rbind한 후 colsum으로 결과 확인해 가장 많이 선택된 10개 변수 선택
  #vars <-order(colSums(final), decreasing = TRUE)[1:d]
  final <- rep(0, p)
  for (i in 1:B1){
    final <- final + apply(max.value[[i]], 2, sum)
  }
  result <- list(output = final/B1)
  #result <- list(output = final/B1, rps <- max.value)
  return(result)
}


set.seed(2022021328)
n <- 400
p <- 1000; d <- 50   # rp

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0.5)
A <- sign(c(x %*% beta + e))

t1 <- 0.442*(1-x1-x2)*A 
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

start_time <- Sys.time()
(obj <- rp_ensemble.2(x, A, R, 500, 500, 50))   # 더 키워야 함: 1000C50? 너무 큼(p=100일 때 B1 = 50 이었으니까 500정도로는 해야할 듯)
end_time <- Sys.time()
end_time - start_time



generate_rp<-function(beta){
  select_beta <- order(beta, decreasing = TRUE)[1:d] #확률 높은 d개 뽑기
  rp<-matrix(0,d,p)
  j=1
  for(i in 1:d){
    if(sum(rp[i,]) == 0){
      rp[i,sort(select_beta)[j]]<-1
      j=j+1
    }
  }
  
  return(rp)
}

# train/test
rp <- generate_rp(obj$output)
z_tr <- x_train %*% t(rp)
classifier <- fit_ITR(z_tr, A_train, R_train, 0.5, 0.1, kern = 'linear')
pred <- predict_ITR(classifier, z_tr) 
sum(pred == A_train)

z_ts <- x_test %*% t(rp)
pred_ts <- predict_ITR(classifier, z_ts)
sum(pred_ts == A_test)



perform_measure <- function(N,d,p){ #N=100
  
  beta_hat_values <- matrix(0, nrow = p, ncol = N)
  CS <- IS <- AC <- numeric(N)
  
  for(i in 1:N){
    #prob는 rp_ensemble 하고 나온 beta_probability vector
    prob <- rp_ensemble(x,A,R,100,50,10,'linear') 
    
    # extract coefficients
    select_beta <- order(prob, decreasing = TRUE)[1:d] #확률 높은 d개 뽑기
    
    beta_hat <- rep(0, p)
    beta_hat[c(select_beta)]<-1  # select 한 변수들 1 넣어주기
    beta_hat_values[,i] <- beta_hat
    
    #CS,IS,AC
    CS[i] <- sum(beta_hat[1:d]==1)
    IS[i] <- sum(beta_hat[1:d]!=1)
    AC[i] <- as.integer((CS[i] == d) & (IS[i] == 0))
    
  }
  
  # MC_MSE, MC_Variance, MC_Bias
  beta_hat_mean <- colMeans(beta_hat_values)
  MC_MSE <- mean(colMeans((beta_hat_values - beta)^2))
  MC_Variance <- mean(colMeans((beta_hat_values - beta_hat_mean)^2))
  MC_Bias <- mean(abs(beta_hat_mean - beta))
  
  results <- data.frame(CS=mean(CS), IS=mean(IS), AC=mean(AC), MC_MSE=MC_MSE, MC_Variance=MC_Variance, MC_Bias=MC_Bias)
  
  return(results)
  
}


# penalized regression
set.seed(2022021328)
n <- 100
p <- 10; d <- 3

data_sampling <- function(n, p, rho){
  n <- n
  p <- p
  
  m <- rep(0,p)
  rho <- rho
  s <- rho^abs(outer(1:p, 1:p,"-"))
  
  x <- rmvnorm(n, m, s)
  
  return(x)
}

x <- data_sampling(n, p, rho = 0.7)

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
A <- sign(c(x %*% beta + e))

t1 <- 0.442*(1-x1-x2)*A 
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

x <- cbind(1, x)

x_train <- x[c(1:80),]
x_test <- x[c(81:100),]
A_train <- A[c(1:80)]
A_test <- A[c(81:100)]
R_train <- R[c(1:80)]
R_test <- R[c(81:100)]

train_index <- sample(1:n, size = 0.8 * n)

x_train <- x[train_index,]
A_train <- A[train_index]
R_train <- R[train_index]

x_valid <- x[-train_index,]
A_valid <- A[-train_index]
R_valid <- R[-train_index]

loss_fun <- function(beta, lambda = 0.1) {
  pred <- x_train %*% beta * (A_train - 0.5)
  loss <- sum((R_train - pred)^2) / n
  penalty <- lambda * sum(abs(beta[-1]))
  return(loss + penalty)
}

beta_start <- rep(0, p+1)
res <- optim(beta_start, loss_fun, method = "BFGS")
(optimal_beta <- res$par)


y.hat <- as.vector(ifelse(x_train %*% optimal_beta > 0, 1, -1))
sum(y.hat == A_train)

y.hat <- as.vector(ifelse(x_test %*% optimal_beta > 0, 1, -1))
sum(y.hat == A_test)




library(plotly)
# simulations setting: linear
n <- 500
p <- 3; d <- 3   # rp

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
#A <- sample(c(-1, 1), n, replace = TRUE, prob = c(0.5, 0.5))
A <- sign(c(x %*% beta + e))

t1 <- 0.442*(1-x1-x2)*A 
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

#A <- as.factor(sample(c(-1, 1), n, replace = TRUE, prob = c(0.5, 0.5)))
A <- as.factor(sign(c(x %*% beta + e)))
data <- data.frame(x1, x2, x3, Q, A)
plot_ly(data, x=~ x2, y=~ x3, z =~ Q, color=~ A, size = 0.3)
plot_ly(data, x=~ x1, y=~ x2, z =~ x3, color=~ A, size =~ Q)

# simulations setting: parabola
n <- 500
p <- 3; d <- 3   # rp

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
A <- sign(c(x %*% beta + e))

t1 <- (x2 - 0.25*x1^2 -1)*A
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

A <- as.factor(sign(c(x %*% beta + e)))
data <- data.frame(x1, x2, x3, Q, A)
plot_ly(data, x=~ x2, y=~ x3, z =~ Q, color=~ A, size = 0.3)
plot_ly(data, x=~ x1, y=~ x2, z =~ x3, color=~ A, size =~ Q)

# simulations setting: ring
n <- 500
p <- 3; d <- 3   # rp

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
A <- sign(c(x %*% beta + e))

t1 <- (0.5-x1^2-x2^2)*(x1^2+x2^2-0.3)*A
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

A <- as.factor(sign(c(x %*% beta + e)))
data <- data.frame(x1, x2, x3, Q, A)
plot_ly(data, x=~ x2, y=~ x3, z =~ Q, color=~ A, size = 0.3)
plot_ly(data, x=~ x1, y=~ x2, z =~ x3, color=~ A, size =~ Q)

# simulations setting: nonlinear
n <- 500
p <- 3; d <- 3   # rp

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- x[,1]; x2 <- x[,2]; x3 <- x[,3]

beta <- c(rep(1, d), rep(0, p-d))
e <- rnorm(n, 0, 0)
A <- sign(c(x %*% beta + e))

t1 <- (1-x1^3 + exp(x3^2 + x2))*A
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t1
R <- rtruncnorm(n, a = 0, mean = Q, sd = 1)

A <- as.factor(sign(c(x %*% beta + e)))
data <- data.frame(x1, x2, x3, Q, A)
plot_ly(data, x=~ x2, y=~ x3, z =~ Q, color=~ A, size = 0.3)
plot_ly(data, x=~ x1, y=~ x2, z =~ x3, color=~ A, size =~ Q)





