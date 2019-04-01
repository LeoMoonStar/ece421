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
    coeff = -(dim/2)*tf.log(2*math.pi*sigma)

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
  N, D = num_pts, dim

  X = tf.placeholder(tf.float32, shape=(N, D), name="X")
  MU = tf.get_variable(name="MU", initializer=tf.random.normal(shape=[K, D]))
  sigma = tf.get_variable(shape=(K, 1), name="sigma")
  pi = tf.get_variable(shape=(1, K), name="pi")

  # Take the expononent of the sigma as per instructions
  sexp = tf.exp(sigma)

  # compute the P(xn | zn = K)
  log_PDF = log_GaussPDF(X, MU, sexp)

  sum = hlp.reduce_logsumexp(hlp.logsoftmax(pi) + log_PDF)

  loss = tf.reduce_sum(-1*sum)

  opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(loss)

  return MU, X, loss, opt, sigma, pi

def MOGLoss(K):
  N, D = num_pts, dim

  X = tf.placeholder(tf.float32, shape=(N, D), name="X_VAL")
  MU = tf.get_variable(name="MU_VAL", initializer=tf.random.normal(shape=[K, D]))
  sigma = tf.get_variable(shape=(K, 1), name="sigma_VAL")
  pi = tf.get_variable(shape=(1, K), name="pi_VAL")

  # Take the expononent of the sigma as per instructions
  sexp = tf.exp(sigma)

  # compute the P(xn | zn = K)
  log_PDF = log_GaussPDF(X, MU, sexp)

  sum = hlp.reduce_logsumexp(hlp.logsoftmax(pi) + log_PDF)

  loss = tf.reduce_sum(-1*sum)

  return MU, X, loss, sigma, pi


def runGmmLoss(K):
  iterations = 300
  MU, X, loss, opt, sigma, pi = MoG(data, K, 0.01)

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
        mu, l, _, s, p = session.run([MU, loss, opt, sigma, pi], feed_dict={X: data})
        #print(val_loss)
        print(mu)
        loss_vec.append(l)

  plotLoss("Loss when K=3", "iterations", "loss", loss_vec)
  #tf.reset_default_graph()

def runGMMClusters(K):
  iterations = 300
  MU, X, loss, opt, sigma, pi = MoG(data, K, 0.01)

  if is_valid:
    MU_val, X_val, val_loss, sigma_val, pi_val  = MOGLoss(val_data, K)
    val_loss_vec = []

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
      mu, l, _, s, p = session.run([MU, loss, opt, sigma, pi], feed_dict={X: data})
      #print(data.shape, K, a.shape)
      if is_valid:
        l_val = session.run([val_loss], feed_dict={X_val: val_data, MU_Val: mu, sigma_val: s, pi_val: p})
        val_loss_vec.append(l_val)
      loss_vec.append(l)

  # Change this so that the posterior log value is used to class the values
  loggy = log_GaussPDF(X, mu, s)
  #print(loggy)
  #print(hlp.logsoftmax(loggy))
  #print(tf.argmin(hlp.logsoftmax(loggy), axis=1))
  cluster_groups = tf.argmin(hlp.logsoftmax(loggy), axis=1)
  print(cluster_groups)
  #group_array = tf.zeros([K])
  #for i in range(0, cluster_groups.shape[0]):
  #  group_array[cluster_groups[i]] += 1
  #print(100*group_array/np.sum(group_array))

  title = "Clusters when K = " + str(K)
  xlabel = "x1"
  ylabel = "x2"

  plotClusters(title, xlabel, ylabel, data, mu, cluster_groups)
  if is_valid:
    plotLoss("Validation Loss when K = " + str(K), "iterations", "loss", loss_vec, val_loss_vec)

  tf.reset_default_graph()

if __name__ == "__main__":
  #runGmmLoss(5)
  runGMMClusters(3)
  plt.show()