#!/usr/bin/env Rscript

rln =function(m, lambda, nu)
{
  # logisitic-normal samples
  # m is number of samples
  # lambda is mean vector
  # nu is covariance matrix
  require(MASS)
  vals = mvrnorm(m, lambda, nu) 
  if (!is.null(dim(vals))){
    vals =  exp(vals)/rowSums(exp(vals))
  }
  else{
    vals =  exp(vals)/sum(exp(vals))
  }
  return(vals)
}

myarg = commandArgs()
outloc = myarg[length(myarg)-1]
ntops = as.integer(myarg[length(myarg)])
m=100 # number of logistic normal samples to use --- determined through aux experiment

lambda = read.table(paste(outloc,'final-lambda.dat', sep=""))
lambda = unlist(lambda)
lambda = matrix(lambda, ncol=ntops, byrow=T)
lambda = lambda[,1:(dim(lambda)[2] - 1)] # remove 0's column
ndoc=dim(lambda)[1]
nu = read.table(paste(outloc,'final-nu.dat', sep=""))
nu = unlist(nu)
nu = matrix(nu, ncol=ntops, byrow=T)
nu = nu[,1:(dim(nu)[2] - 1)]

#hellinger.terms = function(lambda, nu, m)
  # m is number of samples to use
  # lambda is the final lambda matrix -- docs by topics matrix -- remove 0 columns?
  # nu is the final nu matrix -- dimensions?
  sqrt.theta = matrix(0, nrow=dim(lambda)[1], ncol=dim(lambda)[2]);
  for (i in 1:dim(lambda)[1])
  {
    #cat(i,"\n");
    samples = rln(m, lambda[i,], diag(nu[i,])); # logistic normal
    sqrt.theta[i,] = colMeans(sqrt(samples));
  }

igrid = expand.grid(1:ndoc,1:ndoc)
results = apply(igrid,1, function(i) return(2 - 2 * sum(sqrt.theta[i[1],] * sqrt.theta[i[2],])))
results = matrix(results, ndoc, ndoc)
write.table(format(results,digits=3),file=paste(outloc, 'hellinger-docs.csv', sep=''), row.names=F, quote=F, col.names=F)








