import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

is_valid = True

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


# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)

    #given two vectors A(Nx1) B(Nx1) the norm =
    #sqrt((A0 - B0)^2 + ... + (AN - BN)^2) = 
    #sqrt(A0^2 + B0^2 - 2*A0*B0 + ... + AN^2 + BN^2 - 2*AN*BN)

    t1 = -2*tf.matmul(X, MU, transpose_a=False, transpose_b=True)
    sqX = tf.reshape(tf.reduce_sum(tf.square(X), axis=1), [-1, 1])
    sqMU = tf.reshape(tf.reduce_sum(tf.square(MU), axis=1), [1, -1])

    ret = tf.sqrt(tf.maximum(t1 + sqX + sqMU, 0))
    return ret

def computeLoss(X, MU):
  dist_mat = tf.square(distanceFunc(X, MU))
  temp = tf.reduce_max(-1*dist_mat, axis=1)
  temp = -1*temp
  loss = tf.reduce_sum(temp)
  return dist_mat, loss

def kmeansGraph(dataset, K, alpha):
  N = dataset.shape[0]
  D = dataset.shape[1]

  X = tf.placeholder(tf.float32, shape=(N, D), name="X")
  MU = tf.get_variable("MU", initializer=tf.truncated_normal(shape=[K, D]))

  dist_mat, loss = computeLoss(X, MU)

  opt = tf.train.AdamOptimizer(learning_rate=alpha, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
  return dist_mat, MU, X, loss, opt

def kmeansLoss(dataset, K):
  N = dataset.shape[0]
  D = dataset.shape[1]

  X = tf.placeholder(tf.float32, shape=(N, D), name="X_val")
  MU = tf.placeholder(tf.float32, shape=(K, D), name="MU_val")
  _, loss = computeLoss(X, MU)
  return X, MU, loss

def runKmeansLoss(K):
  iterations = 300
  _, MU, X, loss, opt = kmeansGraph(data, K, 0.001)

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
      _, l, mu = session.run([opt, loss, MU], feed_dict={X: data})
        #print(val_loss)
      #print(l)
      loss_vec.append(l)

  plotLoss("Loss when K=3", "iterations", "loss", loss_vec)
  tf.reset_default_graph()

def runKmeansClusters(K):
  iterations = 300
  dist_mat, MU, X, loss, opt = kmeansGraph(data, K, 0.001)

  if is_valid:
    X_val, MU_Val, val_loss = kmeansLoss(val_data, K)
    val_loss_vec = []

  loss_vec = []

  with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    session.run(tf.local_variables_initializer())
    for i in range(0, iterations):
      _, l, mu, a  = session.run([opt, loss, MU, dist_mat], feed_dict={X: data})
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
    plotLoss("Validation Loss when K=" + str(K), "iterations", "loss", loss_vec, val_loss_vec)

  tf.reset_default_graph()


if __name__ == "__main__":
  runKmeansLoss(3)

  clusters = [1, 2, 3, 4, 5]
  for K in clusters:
    runKmeansClusters(K)

  plt.show()