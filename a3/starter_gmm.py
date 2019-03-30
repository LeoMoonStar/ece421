import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

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

def convariance_Mat(X, m):
    # Inputs
    # X: is an NxD matrix
    # m: some of the

    x = tf.math.subtract(X, m)
    print(x)
    mean = tf.reduce_mean(x, axis=0, keep_dims=True)
    cov = tf.matmul(tf.transpose(mean), mean)

    return cov


# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

    # Make a covariance matrix
    cov = convariance_Mat(X, m)

    # Find all the means
    #

    #means = tf.reshape(tf.reduce_sum(tf.square(X), axis=1), [-1, 1])/
    #t1 = -2 * tf.matmul(X, MU, transpose_a=False, transpose_b=True)
    #sqX = tf.reshape(tf.reduce_sum(tf.square(X), axis=1), [-1, 1])
    #sqMU = tf.reshape(tf.reduce_sum(tf.square(MU), axis=1), [1, -1])

    #ret = t1 + sqX + sqMU
    #return ret

#def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    # log_pi: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO

#def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO


mat = [[1, 1], [1, 1]]
mu = [2, 2]

print(convariance_Mat(mat, mu))