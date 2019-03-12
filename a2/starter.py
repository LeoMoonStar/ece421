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
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainTarget, validTarget, testTarget = convertOneHot(trainTarget, validTarget, testTarget)

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

plt.show()

#part 2.2
#Model_Training(trainData, trainTarget)
