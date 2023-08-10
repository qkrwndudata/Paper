indicator.itr <- function(A, x, beta, lambda){
  -(A * x) * (A * x %*% beta < lambda) + 0 * (z > lambda)
}

# loss function -> -value function
solve.itr.gd <- function(X, A, init = NULL, learn_rate, eps = 0.001, max_iter = 100) {
  if (is.null(init)) init <- rep(0, ncol(X))
  beta <- init
  A.hat <- X %*% beta 
  converged = F
  iterations = 0
  while(converged == F) {
    ## Implement the gradient descent algorithm
    new.beta <- beta - learn_rate * apply(A*X *c(indicator.itr(A * X %*% beta, 1)), 2, sum)
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
  obj <- list(est = c(beta), iterations = iterations)
  return(obj)
}

solve.itr.gd.revised <- function(X, A, R, init = NULL, learn_rate, eps = 0.001, max_iter = 100) {
  if (is.null(init)) init <- rep(0, ncol(X))
  beta <- init
  A.hat <- X %*% beta 
  converged = F
  iterations = 0
  while(converged == F) {
    ## Implement the gradient descent algorithm
    new.beta <- beta - learn_rate * apply(A * R * X * c(indicator.itr(A * X %*% beta, 1)), 2, sum)
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
  obj <- list(est = c(beta), iterations = iterations)
  return(obj)
}

fit_ITR_GD <- function(x, A, R, pi, kappa, learn_rate = 0.0001, max_iter = 10000){
  n <- nrow(x)
  eps <- 0.001
  
  K <- poly.kernel(x, param.kernel = 1)
  H <- K * (A %*% t(A))
  
  alpha <- rep(0, n)  # Initialize alpha at 0
  
  converged = F
  iterations = 0
  
  while(converged == F) {
    gradient <- (H %*% alpha) - 1  # Compute the gradient
    new.alpha <- alpha - learn_rate * gradient  # Update alpha
    if (max(abs(new.alpha - alpha)) / max(abs(alpha)) < eps) {
      converged = T
      alpha <- new.alpha
    }
    
    iterations = iterations + 1
    alpha <- new.alpha
    
    if (iterations == max_iter) { 
      converged = T
      warning("Algorithm may not be converged!")
      alpha <- new.alpha
    }
  
  }
  
  alpha <- alpha[,1]
  sv.index <- which(alpha > eps & alpha < kappa * R / pi - eps)
  
  beta <- apply(alpha * A * x, 2, sum) 
  temp <- A[sv.index] - x[sv.index,] %*% beta
  beta0 <- mean(temp)
  
  sol <- list(alpha = alpha, coef = beta, intercept = beta0)
  return(sol)
}


fit_ITR_GD.2 <- function(x, A, R, pi, kappa, learn_rate = 0.0001, max_iter = 10000){
  p <- ncol(x)
  eps <- 0.001
  
  beta <- rep(0, p)  # Initialize alpha at 0
  
  converged = F
  iterations = 0
  
  while(converged == F) {
    gradient <- apply(-(R / ((A * pi) + (1-A)/2)) * A * x, 2, sum)/n  # Compute the gradient
    new.beta <- beta - learn_rate * gradient  # Update alpha
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





predict_ITR.2 <- function(model, newdata){  
    y.hat <- as.vector(ifelse(newdata %*% model$est > 0, 1, -1))
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

(sim <- fit_ITR_GD.2(x, A, R, 0.5, 1, learn_rate = 0.001, max_iter = 1000))
pred <- predict_ITR.2(sim, x)
sum(pred == A)
