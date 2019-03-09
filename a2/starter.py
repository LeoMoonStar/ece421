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
    input = tf.reshape([-1, 28, 28, 1])

    # 3x3 convolution, 1 input, 32 outputs
    w1 =  tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.Variable(tf.random_normal([32]))

    # Unsure if strides is correct/have not added the right [3,3]
    x1 = tf.nn.conv2d(input, w1, strides=[1, 1, 0, 0], padding='SAME')
    print(x1)
    x1 = tf.nn.bias_add(x1, b1)
    print(x1)
    x1 = tf.nn.relu(x1)
    print(x1)

    '''# Convolutional Layer
    #conv1 = tf.keras.layers.Conv2D( inputs=input, filters=32, kernel_size=[3, 3], strides=[1, 1], activation=tf.nn.relu)
    conv1 = tf.nn.conv2d(inputs=input, )
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)'''

    # Batch Normalization layer - axes have not been properly intialized
    #batch1 = tf.keras.layers.BatchNormalization(axis=1, input=conv1)
    mean, variance = tf.nn.moments(x1, axes=[1])
    xn = tf.nn.batch_normalization(x1, mean, variance)


    # Pooling Layer
    #pool1 = tf.layers.max_pooling2d(inputs=batch1, pool_size=[2, 2])
    xp1 = tf.nn.max_pool(xn, ksize=[1, 3, 3, 1], strides=[1, 1, 0, 0], padding='SAME')

    # Flatten Layer (Fix the reshape)
    flat_pool1 = tf.reshape(xp1, [-1, 5 * 5 * 32])
    w2 = tf.get_variable("W2", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([32]))

    '''# Fully Connected Layer 1
    fc1 = tf.keras.layers.Dense(input=flat_pool1, activation=tf.nn.relu, units=784)

    # Fully Connected Layer 2
    fc2 = tf.keras.layers.Dense(input=fc1, activation=tf.nn.softmax, units=10)'''
    fc2 = xp1

    # CE
    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc2)

    return loss

print('hello')
#print(relu(np.array([1,2,3,-2,4])))
#print(softmax(np.array([1,2,3])))
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
#print(trainData.shape)
#print(CE(np.array([1,2,3,-2,4]), np.array([1,2,3,-2,4])))
convolutional_layers(trainData, trainTarget)