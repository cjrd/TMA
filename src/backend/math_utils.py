import numpy as np


def logistic_normal(mean, cov, n):
    """
    returns samples from a logistic normal distribution
    @param mean: mean vector
    @param cov: covariance matrix
    @param n: number of samples
    @return n samples from the logistic normal distribution
    """
    samples = np.random.multivariate_normal(mean,cov, n)
    samples = np.exp(samples)
    return samples/samples.sum(axis=1)[:, np.newaxis]

def hellinger_distance(sqrt_doca, sqrt_docb, axis=1):
    """
    Returns the Hellinger Distance between documents.

    @para doca: is expected to be a 1d array (ie., a single document),
    @param docb: is expected to be a 2d array(ie., the rest of the documents in the
    corpus).

    Note that this expects to be given proper probability distributions.
    """
    return np.sum((sqrt_doca - sqrt_docb)**2, axis=axis)

