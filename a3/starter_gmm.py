import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# Here for testing
is_valid = True

# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

print(num_pts, dim)

def convariance_Mat(X, mean):
    # Inputs
    # X: is an NxD matrix
    # m: some of the

    #mean = tf.reduce_mean(X, axis=0, keep_dims=True)
    # x_aux = tf.subtract(X, mean)
    # cov = tf.matmul(x_aux, x_aux, transpose_a=True)/(x_aux.shape[0])
    # return cov
    return

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)


    # Find all the means
    #

    #means = tf.reshape(tf.reduce_sum(tf.square(X), axis=1), [-1, 1])/
    t1 = -2 * tf.matmul(X, MU, transpose_a=False, transpose_b=True)
    sqX = tf.reshape(tf.reduce_sum(tf.square(X), axis=1), [-1, 1])
    sqMU = tf.reshape(tf.reduce_sum(tf.square(MU), axis=1), [1, -1])

    ret = t1 + sqX + sqMU
    return ret

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    #
    # Outputs:
    # log Gaussian PDF N X K
    #
    # TODO
    pairwise_dist = distanceFunc( X, mu )
    sigma_dist = -1*tf.divide(pairwise_dist, 2*sigma)
    coeff = -1*tf.log(2*math.pi*sigma)

    return coeff + sigma_dist

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    #
    # Outputs
    # log_post: N X K
    #
    # TODO
    p_xz = tf.add( log_PDF, log_pi )

    return logsoftmax(p_xz)


mat = [[1, 1], [1, 1]]
mu = [2, 2]

print(convariance_Mat(mat, mu))