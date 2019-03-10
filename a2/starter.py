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

    return loss

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

def MSE_predict(W, b, x):
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))
    y_hat = np.matmul(X_aux, W_aux) + b
    return y_hat

def CE_predict(W, b, x):
    xn = MSE_predict(W, b, x)
    y_hat = 1 / (1 + np.exp((-1)*xn))
    return y_hat

def accuracy(y_hat, y):
    correct = 0
    for i in range(y.shape[0]):
        if (y_hat[i] > 0.5 and y[i] == 1) or \
           (y_hat[i] < 0.5 and y[i] == 0):
            correct += 1
    return correct/y.shape[0]

def SGDBatches(it, X_d, Y_d, batchSize, reg, lam, sess, train_op, loss):

    data_shape = (batchSize, X_d.shape[1]*X_d.shape[2])
    target_shape = (batchSize, 1)
    weights = None
    bias = None
    a = 0
    for i in range(it):
        x_batch = X_d[i*batchSize:(i+1)*batchSize].reshape(data_shape)
        y_batch = Y_d[i*batchSize:(i+1)*batchSize].reshape(target_shape)
        _, l = sess.run([train_op, loss], feed_dict={X:x_batch, Y:y_batch, lam:reg})
        weights, bias = sess.run([W, b], feed_dict={X:x_batch, Y:y_batch, lam:reg})
        a = accuracy(CE_predict(weights, bias, X_d[i*batchSize:(i+1)*batchSize]), y_batch)

        losses = l

    return losses, a, weights, bias

def SGD(batchSize, iterations, features, labels, beta1=0.9, beta2=0.999, epsilon=1e-04):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
    train_op, loss = Model_Training(features, labels)

    l_train = []
    l_valid = []
    l_test = []
    a_train = []
    a_valid = []
    a_test = []
    trainBatches = int(trainData.shape[0]/batchSize)
    validBatches = int(validData.shape[0]/batchSize)
    testBatches = int(testData.shape[0]/batchSize)
    error_valid = 0
    error_test = 0
    acc_valid = 0
    acc_test = 0
    reg = 0
    lam = 0

    with tf.Session() as sess:
        #sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            i_train = np.arange(trainData.shape[0]); np.random.shuffle(i_train)
            i_valid = np.arange(validData.shape[0]); np.random.shuffle(i_valid)
            i_test = np.arange(testData.shape[0]); np.random.shuffle(i_test)

            trainData = trainData[i_train]
            validData = validData[i_valid]
            testData = testData[i_test]

            trainTarget = trainTarget[i_train]
            validTarget = validTarget[i_valid]
            testTarget = testTarget[i_test]

            losses, accur, weight, bias = SGDBatches(trainBatches, trainData, trainTarget, batchSize, reg, lam, sess,\
                                                     train_op, loss)

            #error_valid = MSE(weight, bias, validData, validTarget, reg)
            #error_test = MSE(weight, bias, testData, testTarget, reg)
            acc_valid = accuracy(MSE_predict(weight, bias, validData), validTarget)
            acc_test = accuracy(MSE_predict(weight, bias, testData), testTarget)

            l_train.append(losses)
            l_valid.append(error_valid)
            l_test.append(error_test)

            a_train.append(accur)
            a_valid.append(acc_valid)
            a_test.append(acc_test)

    iter_plt = range(iterations)

    y1 = [(l_train, 'training', 'r-'), \
         (l_valid, 'valid', 'g-'), \
         (l_test, 'test', 'b-')]

    y2 = [(a_train, 'training', 'r-'), \
         (a_valid, 'valid', 'g-'), \
         (a_test, 'test', 'b-')]

    prefix = "batch size:" + str(batchSize) + ", beta1: " + str(beta1) + ", beta2: " + str(beta2) + ", epsilon: " + str(epsilon)
    name1 = "Training error of " + type + " per epoch with \n" + prefix
    name2 = "accuracy of " + type + " per epoch with \n" + prefix

    plotFigures(name1, "epoch", "error", iter_plt, y1, 2)
    plotFigures(name2, "epoch", "accuracy", iter_plt, y2, 4)
    return

def Model_Training(features, labels):

    loss = convolutional_layers(features, labels)

    opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

    return opt, loss



print('hello')
#print(relu(np.array([1,2,3,-2,4])))
#print(softmax(np.array([1,2,3])))
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
#print(trainData.shape)
#print(CE(np.array([1,2,3,-2,4]), np.array([1,2,3,-2,4])))
print(convolutional_layers(trainData, trainTarget))
