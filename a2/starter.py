import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest


def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def convolutional_layers(features, labels):
    # Input Layer
    input = tf.reshape(features, shape=[-1, 28, 28, 1])

    # 3x3 convolution, 1 input, 32 outputs
    W1 = tf.get_variable("W1", [3, 3, 1, 32], dtype='float64',initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [32], dtype='float64', initializer=tf.contrib.layers.xavier_initializer())
    conv = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + b1, name='conv1')

    # Batch Normalization layer
    mean, variance = tf.nn.moments(conv1, axes=[0, 1, 2])
    bn = tf.nn.batch_normalization(x=conv1, mean=mean, variance=variance, offset=None, scale=None, variance_epsilon=0.001)

    # 2×2 max pooling layer
    pool = tf.nn.max_pool(bn, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Flatten Layer
    pool = tf.reshape(pool, [-1, 6272])

    # Fully connected layer relu
    W2 = tf.get_variable('W2', [6272, 1024], dtype='float64', initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [1024], dtype='float64', initializer=tf.contrib.layers.xavier_initializer())
    fc1 = tf.nn.relu(tf.matmul(pool, W2) + b2)

    # Fully connected layer with softmax
    W3 = tf.get_variable('W3', [1024, 10], dtype='float64', initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [10], dtype='float64', initializer=tf.contrib.layers.xavier_initializer())
    fc2 = tf.matmul(fc1, W3) + b3

    # Loss with cross entropy
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc2)
    loss = tf.reduce_mean(entropy)

    return loss, W3, b3

def plotFigures(title, x_label, y_label, x_arr, y_arr, legend):
    global figure_num
    plt.figure(figure_num)
    plt.ylabel(x_label)
    plt.xlabel(y_label)
    plt.title(title)
    for i in range(len(y_arr)):
        y = y_arr[i][0]
        plot_name = y_arr[i][1]
        color = y_arr[i][2]
        plt.plot(x_arr, y, color, label=plot_name)
    plt.legend(loc=legend)
    figure_num += 1



def Model_Training(features, labels):

    dim = 10
    N = features.shape[0]
    dim_x = features.shape[1]
    dim_y = features.shape[2]
    batch_size = 32
    epoch = 50
    runs = int(N / batch_size)


    loss, W, b = convolutional_layers(features, labels)
    print('hi')
    x_data = features
    y_data = labels


    # Define placeholders to feed mini_batches
    X = tf.placeholder(tf.float32, shape=(batch_size, dim_x*dim_y), name="X")
    Y = tf.placeholder(tf.float32, shape=(batch_size, None), name="Y")

    opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

    init = tf.global_variables_initializer()
    session = tf.Session()
    session.run(init)

    # Fit the line.
    for s in range(epoch):
        for p in range(runs):

            x_batch = x_data[p * batch_size:(p + 1) * batch_size].reshape((batch_size, x_data.shape[1]*x_data.shape[2]))
            y_batch = y_data[p * batch_size:(p + 1) * batch_size]
            print(y_batch.shape)
            _, l = session.run([opt, loss], feed_dict={X: x_batch, Y: y_batch})

            if s % 1 == 0:
                print(s, session.run(W))

        x_data, y_data = shuffle(x_data, y_data)




print('hello')
#print(relu(np.array([1,2,3,-2,4])))
#print(softmax(np.array([1,2,3])))
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
#print(trainData.shape)
#print(CE(np.array([1,2,3,-2,4]), np.array([1,2,3,-2,4])))
Model_Training(trainData, trainTarget)
