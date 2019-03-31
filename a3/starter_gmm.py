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
is_valid = False
# For Validation set
if is_valid:
    valid_batch = int(num_pts / 3.0)
    np.random.seed(45689)
    rnd_idx = np.arange(num_pts)
    np.random.shuffle(rnd_idx)
    val_data = data[rnd_idx[:valid_batch]]
    data = data[rnd_idx[valid_batch:]]

fig_num = 0

def plotLoss(title, xlabel, ylabel, losses, val_losses=[]):
  global fig_num
  plt.figure(fig_num)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.title(title)
  plt.plot(range(len(losses)), losses, label="losses")
  if len(val_losses) > 0:
    plt.plot(range(len(val_losses)), val_losses, label="validation losses")
  plt.legend(loc=0)
  fig_num += 1

def plotClusters(title, xlabel, ylabel, data, centers, data_groups):
    global fig_num
    colors = ["red", "blue", "yellow", "green", "black"]
    data_colors = []
    for i in range(0, data.shape[0]):
        data_colors.append(colors[data_groups[i]])
    plt.figure(fig_num)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    #plt.plot(range(len(losses)), losses, label="losses")
    plt.scatter(data[:,0], data[:,1], c=data_colors)
    plt.scatter(centers[:,0], centers[:,1], c=colors)

    plt.legend(loc=0)
    fig_num += 1

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

    pairwise_dist = distanceFunc( X, mu )
    print(pairwise_dist)
    sigma_dist = -1*tf.div(pairwise_dist, tf.transpose(2*sigma))
    print(sigma_dist)
    coeff = -1*tf.log(2*math.pi*sigma)

    return tf.transpose(coeff) + sigma_dist

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    #
    # Outputs
    # log_post: N X K

    p_xz = tf.add( log_PDF, log_pi )

    return hlp.logsoftmax(p_xz)

def MoG(dataset, K, alpha):
  N = dataset.shape[0]
  D = dataset.shape[1]

  X = tf.placeholder(tf.float32, shape=(N, D), name="X")
  MU = tf.get_variable(name="MU", initializer=tf.random.normal(shape=[K, D]))
  sigma = tf.get_variable(shape=(K, 1), name="sigma")

  # compute the P(xn | zn = K)
  log_PDF = log_GaussPDF(X, MU, sigma)

  # Reduce to a K by 1 matrice containing all of the means
  log_pi = tf.reduce_max(MU, 1)

  # compute the P(z = k)
  p_zk = log_posterior(log_PDF, log_pi)

  loss = p_zk + log_PDF

  opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(-1 * loss)

  return MU, X, loss, opt, sigma


def runGmmLoss(K):
  iterations = 300
  MU, X, loss, opt, sig = MoG(data, K, 0.1)

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
        _, l, mu, s = session.run([opt, loss, MU, sig], feed_dict={X: data})
        #print(val_loss)
        print(s)
        loss_vec.append(l)

  #plotLoss("Loss when K=3", "iterations", "loss", loss_vec)
  #tf.reset_default_graph()



if __name__ == "__main__":
  runGmmLoss(3)
  plt.show()