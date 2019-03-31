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

    pairwise_dist = distanceFunc(X, mu)
    sigma_dist = -1*tf.div(pairwise_dist, tf.transpose(2*sigma))
    coeff = -1*tf.log(2*math.pi*sigma)

    return tf.transpose(coeff) + sigma_dist

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1
    #
    # Outputs
    # log_post: N X K

    p_xz = tf.add(log_PDF, log_pi)

    return hlp.logsoftmax(p_xz)

def MoG(dataset, K, alpha):
  N = dataset.shape[0]
  D = dataset.shape[1]

  X = tf.placeholder(tf.float32, shape=(N, D), name="X")
  MU = tf.get_variable(name="MU", initializer=tf.random.normal(shape=[K, D]))
  sigma = tf.get_variable(shape=(K, 1), name="sigma")
  sexp = tf.exp(sigma)
  pi = tf.get_variable(shape=(1, K), name="pi")

  # compute the P(xn | zn = K)
  log_PDF = log_GaussPDF(X, MU, sexp)

  loss = tf.reduce_sum(-1*(hlp.logsoftmax(pi) + log_PDF))
  #loss = tf.reduce_sum(log_posterior(log_PDF, hlp.logsoftmax(pi)))

  opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

  return MU, X, loss, opt, sigma, pi

def MOGLoss(dataset, K):
  N = dataset.shape[0]
  D = dataset.shape[1]

  X = tf.placeholder(tf.float32, shape=(N, D), name="X")
  Phi = tf.get_variable(name="Phi", initializer=tf.random.normal(shape=[K, D]))
  sigma = tf.get_variable(shape=(K, 1), name="sigma")

  #_, loss = computeLoss(X, MU)
  #return X, MU, loss


def runGmmLoss(K):
  iterations = 300
  MU, X, loss, opt, sig, pi = MoG(data, K, 0.1)

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
        _, l, mu, s, p = session.run([opt, loss, MU, sig, pi], feed_dict={X: data})
        #print(val_loss)
        print(l)
        loss_vec.append(l)

  #plotLoss("Loss when K=3", "iterations", "loss", loss_vec)
  #tf.reset_default_graph()

def runGMMClusters(K):
  iterations = 10000
  MU, X, loss, opt, sig = MoG(data, K, 0.1)

  if is_valid:
    X_val, MU_Val, val_loss  = MOGLoss(val_data, K)
    val_loss_vec = []

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
      _, l, mu, a, s  = session.run([opt, loss, MU, sig], feed_dict={X: data})
      #print(data.shape, K, a.shape)
      if is_valid:
        l_val = session.run([val_loss], feed_dict={X_val: val_data, MU_Val: mu})
        val_loss_vec.append(l_val)
      loss_vec.append(l)

  cluster_groups = np.argmin(a, axis=1)
  group_array = np.zeros((K,))
  for i in range(0, cluster_groups.shape[0]):
    group_array[cluster_groups[i]] += 1
  print(100*group_array/np.sum(group_array))

  title = "Clusters when K = " + str(K)
  xlabel = "x1"
  ylabel = "x2"

  plotClusters(title, xlabel, ylabel, data, mu, cluster_groups)
  if is_valid:
    plotLoss("Validation Loss when K = " + str(K), "iterations", "loss", loss_vec, val_loss_vec)

  tf.reset_default_graph()

if __name__ == "__main__":
  runGmmLoss(3)
  plt.show()