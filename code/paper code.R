##################
### 0. Library ###
##################

library(quantreg)
library(kernlab)
library(truncnorm)
library(svmpath)
library(Matrix)
library(plyr)
library(glmnet)
library(mvtnorm)


################################################################
### 1. Fitting ITR with Gradient Descent for linear function ###
################################################################

fit_ITR_GD.2 <- function(x, A, R, pi, kappa, learn_rate = 0.0001, max_iter = 10000){
  p <- ncol(x)
  eps <- 0.001
  
  beta <- rep(0, p)  # Initialize alpha at 0
  
  converged = F
  iterations = 0
  
  while(converged == F) {
    gradient <- apply(-(R / ((A * pi) + (1-A)/2)) * A * x, 2, sum)/n.  # Compute the gradient
    new.beta <- beta - learn_rate * gradient.  # Update beta
    if (max(abs(new.beta - beta)) / max(abs(beta)) < eps) {
      converged = T
      beta <- new.beta
    }
    
    iterations = iterations + 1
    beta <- new.beta
    
    if (iterations == max_iter) { 
      converged = T
      warning("Algorithm may not be converged!")
      beta <- new.beta
    }
    
  }
  
  obj <- list(est = c(beta))
  return(obj)
}


##############################################################
### 2. Predict function with gradient descent fitted model ###
##############################################################

predict_ITR.2 <- function(model, newdata){  
  y.hat <- as.vector(ifelse(newdata %*% model$est > 0, 1, -1))
  return(y.hat)
}


###################################
### 3. Calculate value function ###
###################################

# Value function 여러 개 만들기
val.func <- function(pred, R, A){ 
  pi.a <- sum(A == 1) / n
  value <- sum(R * (A == pred) / pi.a)
  return(value/n)
}


####################################################
### 4. Define Random Projection generate function ###
####################################################

generate_projection <- function(d, p){
  a <- matrix(0, p, d)   # Initialize matrix
  sel <- sample(1:p, d, replace = F)
  for (i in 1:d){
    a[sel[i], i] <- 1
  }
  return(t(a))
}


#############################################################
### 5. Variable Screening with Random Projection Ensemble ###
#############################################################

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


###############################################################################
### 6. define final random projection matrix from variable screening method ###
###############################################################################

generate_rp <- function(beta){
  select_beta <- order(beta, decreasing = TRUE)[1:d] #확률 높은 d개 뽑기
  rp <- matrix(0, d, p)
  j <- 1
  for(i in 1:d){
    if(sum(rp[i,]) == 0){
      rp[i, sort(select_beta)[j]] <- 1
      j <- j + 1
    }
  }
  
  return(rp)
}


#####################################
### 7. Define performance measure ###
#####################################

perform_measure <- function(N, d, p){   # N = 100
  
  beta_hat_values <- matrix(0, nrow = p, ncol = N)
  CS <- IS <- AC <- numeric(N)
  
  for(i in 1:N){
    #prob는 rp_ensemble 하고 나온 beta_probability vector
    prob <- rp_ensemble(x, A, R, 100, 50, 10, 'linear') 
    
    # extract coefficients
    select_beta <- order(prob, decreasing = TRUE)[1:d] #확률 높은 d개 뽑기
    
    beta_hat <- rep(0, p)
    beta_hat[c(select_beta)] <- 1  # select 한 변수들 1 넣어주기
    beta_hat_values[,i] <- beta_hat
    
    #CS,IS,AC
    CS[i] <- sum(beta_hat[1:d] ==1 )
    IS[i] <- sum(beta_hat[1:d] != 1)
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


#############################
### 8. Simulation Setting ###
#############################

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
(obj <- rp_ensemble.2(x, A, R, 500, 500, 50))   # 더 키워야 함: 1000C50? 너무 큼 (p=100일 때 B1 = 50 이었으니까 500 정도로는 해야할 듯)
end_time <- Sys.time()
end_time - start_time

# train/test split
x_train <- x[c(1:320),]
x_test <- x[c(321:400),]
A_train <- A[c(1:320)]
A_test <- A[c(321:400)]
R_train <- R[c(1:320)]
R_test <- R[c(321:400)]

rp <- generate_rp(obj$output)
z_tr <- x_train %*% t(rp)
classifier <- rp_ensemble.2(z_tr, A_train, R_train, 0.5, 0.1)
pred <- predict_ITR.2(classifier, z_tr) 
sum(pred == A_train)

z_ts <- x_test %*% t(rp)
pred_ts <- predict_ITR.2(classifier, z_ts)
sum(pred_ts == A_test)


#############################
### 9. Marginal Screening ###
#############################

MarginalScreen <- function(x, A, R){
  p <- ncol(x); n <- nrow(x)
  k <- n / log(n)
  values <- list()
  for (i in 1:p){
    new.x <- as.matrix(x[, i])
    classifier <- fit_ITR_GD.2(new.x, A, R, 0.5, 1, learn_rate = 0.001, max_iter = 1000)
    pred <- predict_ITR.2(classifier, new.x)
    value <- val.func(pred, R, A)
    
    values[[i]] <- value
  }
  values <- as.matrix(unlist(values), ncol = 1)
  colnames(values) <- c("value")
  values <- as.data.frame(values)
  final <- order(values$value, decreasing = T)[1:k]
  return(x[, final])
}

MarginalScreen(x, A, R)
