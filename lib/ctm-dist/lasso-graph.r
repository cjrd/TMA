# Graphical model selection using the Lasso, as
# proposed by Meinshausen and Buhlmann

# April, 2007 -- Dave Blei and John Lafferty
#
# To apply this to topic graphs, we take the variational means
# (lambda) for each document, and treat these as data.  We then
# regress each variable (topic) onto the others using the lasso, and
# consider the indices of the non-zero entries as estimates of the
# neighbors of the node in the inverse covariance.  The graph is then
# formed by including an edge if either/both (OR/AND) of the endpoints
# include it in the corresponding penalized regression.

library(lasso2)
# it's possible to use the lars package as well, with some minor mods

# Inputs
#   file:   n x p data matrix -- e.g., the variational means ("final-lambda.dat")
#   lambda: relative bound on the l1-norm of the parameters, in [0,1]
#   and=T:  if and=T/F then the graph is computed by taking the intersction/union of the nbhds
#
# Output
#   Ihat:   matrix of 0/1, with 1 indicating an edge in the graph
lam = read.table('/Users/cradreed/Research/TMA/tmaout/tmp3sVK1F_formdata/ctm/final-lambda.dat')
lam = unlist(lam) 
x = matrix(lam, nrow=2000, ncol=10, byrow=T)
x = x[,1:9]
lambda = 0.3
and = T
# build.graph = function(x, lambda, and=T) {
  x = scale(x)
  p = ncol(x)
  n = nrow(x)
  Shat = matrix(F,p,p)

  cat("n=",n," p=",p, " lambda=",lambda,"\n", sep="")
  for (j in 1:p) {
    cat(".")
    if (j %% 10 == 0) {
      cat(j)
    }
    # The response is the j-th column
    y = x[,j]
    X = x[,-j]

    # Do the l1-regularized regression
    # Note: the bound in l1ce code is the upper bound on the l1
    # norm.  So, a larger bound is a weaker constraint on the model
    data = data.frame(cbind(y,X))
    out = l1ce(y ~ X, data=data, sweep.out = ~1, bound=lambda)

    indices = (1:p)[-j]
    beta = coef(out)[2:p] # skipping the intercept
    nonzero = indices[beta > 0]
    Shat[j,nonzero] = T
    Shat[j,j] = T
  }
  cat("\n")

  # Include an edge if either (and=F) or both (and=T) endpoints are neighbors
  Ihat = matrix(F,p,p)
  if (and) {
    for (i in 1:p) {
      Ihat[,i] = Shat[,i] & Shat[i,]
    }
  }else
  {
    for (i in 1:p) {
      Ihat[,i] = Shat[,i] | Shat[i,]
    }      
  }
  image(Ihat,col=heat.colors(2),xaxp=c(-1,2,1),yaxp=c(-1,2,1))
  title(main = "Estimated graph")
  #return(Ihat)


