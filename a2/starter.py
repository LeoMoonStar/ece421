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


def relu(x):
    zeros = np.zeros(x.shape)
    return np.maximum(x, zeros)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x/sum_exp_x


def computeLayer(X, W, b):
    return W@X + b

#assuming target is a one hot vector pertaining to which class it 
#corresponds to
def CE(target, prediction):
    N = target.shape[0]

    #Apply softmax to the prediction values
    # Is this getting the average though or just the CE?
    sp = np.apply_along_axis(softmax, 0, prediction)
    log_soft = np.log(sp)
    return -1*np.sum(target*log_soft)/N

def gradCE(target, prediction):
    N = target.shape[0]
    np.apply_along_axis(softmax, 0, prediction)
    return -1*np.sum(target*(1/prediction))

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

def convolutional_layers(features, labels):
    # Input Layer
    input = tf.reshape(features, shape=[-1, 28, 28, 1])

    # 3x3 convolution, 1 input, 32 outputs
    W1 = tf.get_variable("W1", [3, 3, 1, 32], dtype='float64',initializer=tf.contrib.layers.xavier_initializer())
    print(W1)
    b1 = tf.get_variable('b1', [32], dtype='float64', initializer=tf.contrib.layers.xavier_initializer())
    print(b1)
    conv = tf.nn.conv2d(input, W1, strides=[1, 1, 1, 1], padding='SAME')
    print(conv)
    conv1 = tf.nn.relu(conv + b1, name='conv1')
    print(conv1)

    '''# Convolutional Layer
    #conv1 = tf.keras.layers.Conv2D( inputs=input, filters=32, kernel_size=[3, 3], strides=[1, 1], activation=tf.nn.relu)
    conv1 = tf.nn.conv2d(inputs=input, )
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)'''

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

    return loss

print('hello')
#print(relu(np.array([1,2,3,-2,4])))
#print(softmax(np.array([1,2,3])))
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
#print(trainData.shape)
#print(CE(np.array([1,2,3,-2,4]), np.array([1,2,3,-2,4])))
print(convolutional_layers(trainData, trainTarget))
