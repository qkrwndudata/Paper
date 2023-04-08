## itr linear classifier를 풀기 위한 gradient descent 알고리즘 구현 코드 입니다.

# hinge loss 미분 문제를 풀기 위한 subgradient 함수 생성 
indicator.itr <- function(z, lambda){
  (z < lambda) + 0 * (z > lambda)
}

# gradient descent 구현
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
