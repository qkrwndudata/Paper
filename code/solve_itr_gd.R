indicator.itr <- function(z, lambda){
  (z < lambda) + 0 * (z > lambda)
}

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

# setting 1: ordinal
set.seed(1)
n <- 100
p <- 3
x <- matrix(rnorm(n*p), n, p)
X <- cbind(rep(1, n), x)
beta <- rep(1, p+1)
A <- sample(c(-1, 1), 100, replace = TRUE, prob = c(1/2, 1/2))

solve.itr.gd(X, R, learn_rate = 0.01)


# setting 2: ITR
n <- 100; p <- 50

x <- matrix(runif(n * p, -1, 1), n, p)
x1 <- X[,1]; x2 <- X[,2]; x3 <- X[,3]

A <- sample(c(-1, 1), n, replace = TRUE, prob = c(0.5, 0.5))

t <- 0.442*(1-x1-x2)*A
Q <- 1 + 2*x1 + x2 + 0.5*x3 + t
R <- rnorm(n, mean = Q, sd = 1)

solve_ITR(X, A, R, 0.5, 0.1)

