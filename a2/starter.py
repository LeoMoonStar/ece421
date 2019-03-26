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


def relu(x):
    zeros = np.zeros(x.shape)
    return np.maximum(x, zeros)

def gradRelu(x):
    zeros = np.zeros(x.shape)
    x_aux = np.maximum(x, zeros)
    return (x_aux > 0).astype(int)

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1).reshape((x.shape[0], 1))
    return exp_x/sum_exp_x


def computeLayer(X, W, b):
    return X@W + b

#assuming target is a one hot vector pertaining to which class it
#corresponds to
def CE(target, prediction):
    N = target.shape[0]
    log_soft = np.log(prediction)

    return -1*np.sum(target*log_soft)/N

def gradCE(target, prediction):
    N = target.shape[0]
    return np.sum(target/prediction, axis=0)/N

def accuracy(target, prediction):
    N = target.shape[0]
    max_vals = prediction.max(axis=1, keepdims=1) == prediction
    max_vals = max_vals.astype(int)
    return np.sum(max_vals*target)/N

def forward_prop(X, W1, b1, W2, b2, y):
    S1 = computeLayer(X, W1, b1)
    X1 = relu(S1)
    S2 = computeLayer(X1, W2, b2)
    s_X2 = softmax(S2)
    return s_X2, X1, S1

def NN_train(X_aux, y, validData, validTarget, testData, testTarget, epochs, hidden_units):
    num_train_ex = X_aux.shape[0]
    num_features = X_aux.shape[1]*X_aux.shape[2]

    #initialize data matrix
    X = X_aux.reshape((num_train_ex, num_features))
    X_v = validData.reshape(validData.shape[0], num_features)
    X_te = testData.reshape(testData.shape[0], num_features)

    mu, var = 0, 2/(num_features + hidden_units)
    std = var**(1/2)
    W1 = np.random.normal(mu, std, (num_features, hidden_units))
    v_W1 = np.full(W1.shape, 1*10**(-5))

    mu, var = 0, 2/(hidden_units + 10)
    std = var**(1/2)
    W2 = np.random.normal(mu, std, (hidden_units, 10))
    v_W2 = np.full(W2.shape, 1*10**(-5))

    mu, var = 0, 2/(1 + hidden_units)
    std = var**(1/2)
    b1 = np.random.normal(mu, std, (1, hidden_units))
    v_b1 = np.full(b1.shape, 1*10**(-5))

    mu, var = 0, 2/(1 + 10)
    std = var**(1/2)
    b2 = np.random.normal(mu, std, (1, 10))
    v_b2 = np.full(b2.shape, 1*10**(-5))

    #velocity
    gamma = 0.9
    alpha = 0.01

    #for plotting
    errors_tr = []
    acc_tr = []
    errors_v = []
    acc_v = []
    errors_te = []
    acc_te = []

    for i in range(0, epochs):
        #forward propagation
        s_X2, _, _ = forward_prop(X_v, W1, b1, W2, b2, validTarget)
        E_v = CE(validTarget, s_X2)
        A_v = accuracy(validTarget, s_X2)

        s_X2, _, _ = forward_prop(X_te, W1, b1, W2, b2, testTarget)
        E_te = CE(testTarget, s_X2)
        A_te = accuracy(testTarget, s_X2)

        s_X2, X1, S1 = forward_prop(X, W1, b1, W2, b2, y)
        E_tr = CE(y, s_X2)
        A_tr = accuracy(y, s_X2)

        #back propagation
        d1_n = (s_X2-y)
        d1 = np.sum(d1_n, axis=0)/num_train_ex
        v_b2 = gamma*v_b2 + alpha*d1

        v_W2 = gamma*v_W2 + alpha*((d1_n.T@X1).T/num_train_ex)

        d2_n = d1_n@W2.T*gradRelu(S1)
        d2 = np.sum(d2_n, axis=0)/num_train_ex
        v_b1 = gamma*v_b1 + alpha*d2

        v_W1 = gamma*v_W1 + alpha*((d2_n.T@X).T/num_train_ex)

        #Gradient descent
        b2 = b2 - v_b2
        W2 = W2 - v_W2
        b1 = b1 - v_b1
        W1 = W1 - v_W1

        errors_tr.append(E_tr)
        acc_tr.append(A_tr)
        errors_v.append(E_v)
        acc_v.append(A_v)
        errors_te.append(E_te)
        acc_te.append(A_te)

    return errors_tr, acc_tr, errors_v, acc_v, errors_te, acc_te

figure_num = 1

def plotFigures(title, x_label, y_label, x_arr, y_arr, legend):
    global figure_num
    plt.figure(figure_num)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    for i in range(len(y_arr)):
        y = y_arr[i][0]
        plot_name = y_arr[i][1]
        color = y_arr[i][2]
        plt.plot(x_arr, y, color, label=plot_name)
    plt.legend(loc=legend)
    figure_num += 1

def shuffle(trainData, trainTarget):
    np.random.seed(421)
    randIndx = np.arange(len(trainData))
    target = trainTarget
    np.random.shuffle(randIndx)
    data, target = trainData[randIndx], target[randIndx]
    return data, target


def convolutional_layers(features, labels, lam, dropout):
    # Input Layer
    input = tf.reshape(features, shape=[-1, 28, 28, 1])

    # 3x3 convolution, 1 input, 32 outputs
    W1 = tf.get_variable("W1", [3, 3, 1, 32], dtype='float32',initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [32], dtype='float32', initializer=tf.contrib.layers.xavier_initializer())
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
    W2 = tf.get_variable('W2', [6272, 784], dtype='float32', initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [784], dtype='float32', initializer=tf.contrib.layers.xavier_initializer())
    if dropout > 0:
        dropout = tf.nn.dropout(pool,keep_prob=0.9)
        fc1 = tf.nn.relu(tf.matmul(dropout, W2) + b2)
    else:
        fc1 = tf.nn.relu(tf.matmul(pool, W2) + b2)

    # Fully connected layer with softmax
    W3 = tf.get_variable('W3', [784, 10], dtype='float32', initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [10], dtype='float32', initializer=tf.contrib.layers.xavier_initializer())
    fc2 = tf.matmul(fc1, W3) + b3
    sm = tf.nn.softmax(fc2)
    acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(sm, 1), predictions=tf.argmax(labels, 1))

    # Loss with cross entropy
    regu = lam*(tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))

    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fc2)
    loss = tf.reduce_mean(entropy) + regu

    return loss, W3, b3, acc_op

def Model_Training(features, labels, lam, dropout):

    dim = 10
    N = features.shape[0]
    dim_x = features.shape[1]
    dim_y = features.shape[2]
    batch_size = 32
    epoch = 50
    runs = int(N / batch_size)

    # Define placeholders to feed mini_batches
    X = tf.placeholder(tf.float32, shape=(batch_size, dim_x * dim_y), name="X")
    Y = tf.placeholder(tf.float32, shape=(batch_size, None), name="Y")

    loss, W, b, accer = convolutional_layers(X, Y, lam, dropout)

    opt = tf.train.AdamOptimizer(0.0001).minimize(loss)

    return W, b, Y, X, loss, opt, accer

def Model_Batches(it, x_data, y_data, batch_size, session, X, Y, opt, loss, accer, type):
    avg_acc = 0
    avg_loss = 0
    
    for p in range(it):
        x_batch = x_data[p * batch_size:(p + 1) * batch_size].reshape((batch_size, x_data.shape[1] * x_data.shape[2]))
        y_batch = y_data[p * batch_size:(p + 1) * batch_size]
        if type == 'train':
            _, l, acc = session.run([opt, loss, accer], feed_dict={X: x_batch, Y: y_batch})
        else:
            l, acc = session.run([loss, accer], feed_dict={X: x_batch, Y: y_batch})
        avg_acc += acc
        avg_loss += l
    print(l, acc)

    avg_acc /= it
    avg_loss /= it
    return avg_loss, avg_acc

def BuildGraphs(batchSize, iterations, lam, dropout):

    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

    W, b, Y, X, loss, train_op, accer = Model_Training(trainData, trainTarget, lam, dropout)

    l_train = []
    l_valid = []
    l_test = []
    a_train = []
    a_valid = []
    a_test = []
    trainBatches = int(trainData.shape[0] / batchSize)
    validBatches = int(validData.shape[0] / batchSize)
    testBatches = int(testData.shape[0] / batchSize)


    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        #sess.run(tf.global_variables_initializer())
        for i in range(iterations):
            i_train = np.arange(trainData.shape[0])
            np.random.shuffle(i_train)
            i_valid = np.arange(validData.shape[0])
            np.random.shuffle(i_valid)
            i_test = np.arange(testData.shape[0])
            np.random.shuffle(i_test)

            trainData = trainData[i_train]
            validData = validData[i_valid]
            testData = testData[i_test]

            trainTarget = trainTarget[i_train]
            validTarget = validTarget[i_valid]
            testTarget = testTarget[i_test]

            error_train, acc_train = Model_Batches(trainBatches, trainData, trainTarget, \
                          batchSize, session, X, Y, train_op, loss, accer, 'train')
            error_valid, acc_valid = Model_Batches(validBatches, validData, validTarget, \
                          batchSize, session, X, Y, train_op, loss, accer, 'valid')
            error_test, acc_test = Model_Batches(testBatches, testData, testTarget, \
                          batchSize, session, X, Y, train_op, loss, accer, 'test')
            #error_valid, acc_valid = session.run([loss, accer], feed_dict={X: validData, Y: validTarget})
            #error_test, acc_test = session.run([loss, accer], feed_dict={X: testData, Y: testTarget})

            l_train.append(error_train)
            l_valid.append(error_valid)
            l_test.append(error_test)

            a_train.append(acc_train)
            a_valid.append(acc_valid)
            a_test.append(acc_test)

        iter_plt = range(iterations)
        y1 = [(l_train, 'training', 'r-'), \
              (l_valid, 'valid', 'g-'), \
              (l_test, 'test', 'b-')]
        y2 = [(a_train, 'training', 'r-'), \
              (a_valid, 'valid', 'g-'), \
              (a_test, 'test', 'b-')]

        if lam > 0:
            prefix = "lambda : " + str(lam)
            name1 = "Training error of network with \n" + prefix
            name2 = "Accuracy of network with \n" + prefix
        elif dropout > 0:
            prefix = "Dropout percentage : " + str(dropout)
            name1 = "Training error of network with \n" + prefix
            name2 = "Accuracy of network with \n" + prefix
        else:
            prefix = "batch size: 32"
            name1 = "Training error of network with \n" + prefix
            name2 = "Accuracy of network with \n" + prefix

        plotFigures(name1, "epoch", "error", iter_plt, y1, 2)
        plotFigures(name2, "epoch", "accuracy", iter_plt, y2, 4)


trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)
'''
#error for vanilla NN training
epoch = 200
hidden_units = 1000

train_error, train_acc, valid_error, valid_acc, test_error, test_acc = \
    NN_train(trainData, trainTarget, validData, validTarget, testData, testTarget, epoch, hidden_units)

x_label = "epochs"
y_label = "CE (cross entropy error)"
title = "CE over epochs"
y_err = [(train_error, 'training', 'r-'),(valid_error, 'valid', 'g-'),(test_error, 'test', 'b-')]

plotFigures(title, x_label, y_label, range(epoch), y_err, 1)

y_label = "Accuracy"
title = "Accuracy over epochs"
y_acc = [(train_acc, 'training', 'r-'),(valid_acc, 'valid', 'g-'),(test_acc, 'test', 'b-')]
plotFigures(title, x_label, y_label, range(epoch), y_acc, 4)


#part 1.4; 100 hidden units
epoch = 200
hidden_units = 100

test_acc100 = []
_, _, _, _, _, test_acc100 = \
    NN_train(trainData, trainTarget, validData, validTarget, testData, testTarget, epoch, hidden_units)

#part 1.4; 500 hidden units
epoch = 200
hidden_units = 500

test_acc500 = []
_, _, _, _, _, test_acc500 = \
    NN_train(trainData, trainTarget, validData, validTarget, testData, testTarget, epoch, hidden_units)

#part 1.4; 2000 hidden units
epoch = 200
hidden_units = 2000

test_acc2000 = []
_, _, _, _, _, test_acc2000 = \
    NN_train(trainData, trainTarget, validData, validTarget, testData, testTarget, epoch, hidden_units)

y_label = "Accuracy"
title = "Accuracy of test sets using different hidden units"
y_acc = [(test_acc100, '100 hidden units', 'r-'),(test_acc500, '500 hidden units', 'g-'),(test_acc2000, '2000 hidden units', 'b-')]
plotFigures(title, x_label, y_label, range(epoch), y_acc, 4)
'''

#regular run
lam = 0
dropout = 0
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()

#lambda = 0.01
lam = 0.01
dropout = 0
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()

#lambda = 0.1
lam = 0.1
dropout = 0
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()

#lambda = 0.5
lam = 0.5
dropout = 0
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()

#dropout = 0.9
lam = 0
dropout = 0.9
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()


#dropout = 0.75
lam = 0
dropout = 0.75
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()

#dropout = 0.5
lam = 0
dropout = 0.5
BuildGraphs(32, 50, lam, dropout)
tf.reset_default_graph()

plt.show()