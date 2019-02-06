import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

figure_num = 1
#Temp
# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

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

def LeastSquares(x, y):
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    X_aux = x.reshape((num_train_ex, num_pixels))

    X_aux_t = np.transpose(X_aux)
    inverse = np.linalg.inv(np.matmul(X_aux_t, X_aux))
    W = np.matmul(np.matmul(inverse, X_aux_t), y)
    return W.reshape((x.shape[1], x.shape[2]))

def MSE(W, b, x, y, reg):
    #reshaping data and weights
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))

    #calculating cost
    WX = np.matmul(X_aux, W_aux)
    cost = WX + b - y
    cost = np.sum(cost*cost)/(2*num_train_ex)

    #regularization
    regu = np.matmul(W_aux.transpose(), W_aux)*reg/2

    return np.sum(cost+regu)

#calculates gradient of MSE function with respect to
#W and b
def gradMSE(W, b, x, y, reg):
    #reshaping data and weights
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))

    #calculating gradient of weights
    e_in = np.matmul(X_aux, W_aux) + b - y
    grad_W = np.matmul(X_aux.transpose(), e_in)/num_train_ex + reg*W_aux

    #calculating gradient of bias
    grad_b = np.sum(e_in)/num_train_ex

    return grad_W.reshape((x.shape[1], x.shape[2])), grad_b

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))

    xn = np.matmul(X_aux, W_aux) + b
    sigma = 1 / (1 + np.exp((-1)*xn))

    cross_entropy = np.sum(((-1)*y*np.log(sigma) - (1 - y)*np.log(1-sigma)))
    cross_entropy /= num_train_ex

    #regularization
    regu = np.matmul(W_aux.transpose(), W_aux)*reg/2

    return np.sum(cross_entropy+regu)

def gradCE(W, b, x, y, reg):
    # Your implementation here
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))

    xn = np.matmul(X_aux, W_aux) + b

    sigma = (1 / (1 + np.exp(xn)))

    e_in = (-1)*y*sigma + (1 - y)*sigma*np.exp(xn)
    grad_W = np.matmul(X_aux.transpose(), e_in)/num_train_ex + reg*W_aux
    #print(grad_W)

    grad_b = np.sum(e_in)/num_train_ex

    return grad_W.reshape((x.shape[1], x.shape[2])), grad_b

def accuracy(W, x, b, y):
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))

    y_hat = np.matmul(X_aux, W_aux) + b
    correct = 0
    for i in range(y.shape[0]):
        if (y_hat[i] > 0.5 and y[i] == 1) or \
           (y_hat[i] < 0.5 and y[i] == 0):
            correct += 1
    return correct/y.shape[0]

def grad_descent(W, b, trainingData, trainingLabels, validData, validLabels, testData, testLabels, alpha, iterations, reg, EPS, lossType="None"):
    i = 0
    iter_plt = []
    error_train_plt = []
    error_valid_plt = []
    error_test_plt = []
    acc_train_plt = []
    acc_valid_plt = []
    acc_test_plt = []

    error1 = float("inf")
    error2 = 0
    error_valid = 0
    error_test = 0
    acc_train = 0
    acc_valid = 0
    acc_test = 0

    b_grad = None
    W_grad = None
    while i < iterations and abs(error1 - error2) >= EPS:
        error1 = error2

        if lossType == "MSE":
            error2 = MSE(W, b, trainingData, trainingLabels, reg)
            error_valid = MSE(W, b, validData, validLabels, reg)
            error_test = MSE(W, b, testData, testLabels, reg)

            W_grad, b_grad = gradMSE(W, b, trainingData, trainingLabels, reg)
        elif lossType == "CE":
            error2 = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)
            error_valid = crossEntropyLoss(W, b, validData, validLabels, reg)
            error_test = crossEntropyLoss(W, b, testData, testLabels, reg)

            W_grad, b_grad = gradCE(W, b, trainingData, trainingLabels, reg)

        acc_train = accuracy(W, trainingData, b, trainingLabels)
        acc_valid = accuracy(W, validData, b, validLabels)
        acc_test = accuracy(W, testData, b, testLabels)

        W = W - alpha*W_grad
        b = b - alpha*b_grad

        iter_plt.append(i)
        error_train_plt.append(error2)
        error_valid_plt.append(error_valid)
        error_test_plt.append(error_test)

        acc_train_plt.append(acc_train)
        acc_valid_plt.append(acc_valid)
        acc_test_plt.append(acc_test)
        if i % 1000 == 0:
            print(error2)
        i += 1


    '''y1 = [(error_train_plt, 'training', 'r-'), \
         (error_valid_plt, 'valid', 'g-'), \
         (error_test_plt, 'test', 'b-')]
    y2 = [(acc_train_plt, 'training', 'r-'), \
         (acc_valid_plt, 'valid', 'g-'), \
         (acc_test_plt, 'test', 'b-')]
    name1 = "Training error of " + lossType + " per epoch"
    name2 = "accuracy of " + lossType + " per epoch"

    plotFigures(name1, "epoch", "error", iter_plt, y1, 2)
    plotFigures(name2, "epoch", "accuracy", iter_plt, y2, 4)
    return W, b'''

    return error_train_plt, error_valid_plt, error_test_plt, acc_train_plt, acc_valid_plt, acc_test_plt, iter_plt

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    dim_x = trainData.shape[1]
    dim_y = trainData.shape[2]

    tf.set_random_seed(421)
    W_shape = (dim_x*dim_y, 1)
    W = tf.get_variable("W", initializer=tf.truncated_normal(shape=W_shape, stddev=0.5))
    b = tf.get_variable("b", initializer=tf.truncated_normal(shape=[], stddev=0.5))

    X = tf.placeholder(tf.float32, shape=(500, dim_x*dim_y), name="X")
    Y = tf.placeholder(tf.float32, shape=(500, None), name="Y")
    lam = tf.placeholder(tf.float32, shape=None, name="lam")

    predict = None
    loss = None
    if lossType == "MSE":
        predict = tf.matmul(X, W) + b
        loss = tf.losses.mean_squared_error(labels=Y, predictions=predict)
    elif lossType == "CE":
        logit = -1*(tf.matmul(X, W) + b)
        predict = tf.sigmoid(logit)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit))
    #print(loss.shape)
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss + lam/2*tf.matmul(tf.transpose(W), W))
    acc = tf.count_nonzero(tf.greater(predict, 0.5))/predict.shape[0]
    #acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(Y, 1), predictions=tf.argmax(predict, 1))

    return W, b, predict, Y, X, loss, train_op, acc, lam

def SGDBatches(it, X_d, Y_d, batchSize, lam, sess, X, Y, W, b, train_op, loss, acc):
    #print(X_d.shape[0])
    data_shape = (batchSize, X_d.shape[1]*X_d.shape[2])
    target_shape = (batchSize, 1)
    losses = []
    accuracies = []

    for i in range(it):
        x_batch = X_d[i*batchSize:(i+1)*batchSize].reshape(data_shape)
        y_batch = Y_d[i*batchSize:(i+1)*batchSize].reshape(target_shape)
        _, l = sess.run([train_op, loss], feed_dict={X:x_batch, Y:y_batch, lam:0})
        weights, bias = sess.run([W, b], feed_dict={X:x_batch, Y:y_batch, lam:0})
        a = accuracy(weights, X_d[i*batchSize:(i+1)*batchSize], bias, y_batch)
        losses.append(l)
        accuracies.append(a)
        # _, l = sess.run([train_op, loss], feed_dict={X:x_batch, Y:y_batch})
        # l = 0
        # losses.append(l)

    return losses, accuracies

def SGD(batchSize, iterations, type):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W, b, predict, Y, X, loss, train_op, acc, lam = buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-08, lossType=type, learning_rate=0.001)

    l_train = []
    l_valid = []
    l_test = []
    a_train = []
    a_valid = []
    a_test = []
    trainBatches = int(trainData.shape[0]/batchSize)
    validBatches = int(validData.shape[0]/batchSize)
    testBatches = int(testData.shape[0]/batchSize)

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

            l_t, a_t = \
                SGDBatches(trainBatches, trainData, trainTarget, batchSize, lam, \
                    sess, X, Y, W, b, train_op, loss, acc)
            l_v, a_v = \
                SGDBatches(validBatches, validData, validTarget, batchSize, lam, \
                    sess, X, Y, W, b, train_op, loss, acc)
            l_te, a_te = \
                SGDBatches(testBatches, testData, testTarget, batchSize, lam, \
                    sess, X, Y, W, b, train_op, loss, acc)

            l_train.append(l_t[-1])
            # l_valid.append(l_v[-1])
            # l_test.append(l_te[-1])

            a_train.append(a_t[-1])
            # a_valid.append(a_v[-1])
            # a_test.append(a_te[-1])

    iter_plt = range(iterations)
    #print(len(l_train), l_train)
    y1 = [(l_train, 'training', 'r-')]
    # y1 = [(l_train, 'training', 'r-'), \
    #      (l_valid, 'valid', 'g-'), \
    #      (l_test, 'test', 'b-')]
    y2 = [(a_train, 'training', 'r-')]
    # y2 = [(a_train, 'training', 'r-'), \
    #      (a_valid, 'valid', 'g-'), \
    #      (a_test, 'test', 'b-')]
    name1 = "Training error of " + type + " per epoch"
    name2 = "accuracy of " + type + " per epoch"

    plotFigures(name1, "epoch", "error", iter_plt, y1, 2)
    plotFigures(name2, "epoch", "accuracy", iter_plt, y2, 4)
    return

def main():
    #plotting
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    weights = np.zeros((trainData.shape[1], trainData.shape[2]))
    bias = 0
    iterations = 5000

    #####
    #1.3#
    #####
    '''alpha = 0.005
    lam = 0
    W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    alpha = 0.001
    lam = 0
    W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    alpha = 0.0001
    lam = 0
    W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")'''

    #####
    #1.3 - seperate graphs on Alphas#
    #####
    '''alpha = 0.005
    lam = 0
    train1, valid1, test1, atrain1, avalid1, atest1, iter_plt = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    name1 = "Error of training data using MSE per epoch"
    name2 = "Error of valid data using MSE per epoch"
    name3 = "Error of test data using MSE per epoch"
    name4 = "Accuracy of MSE using training data per epoch"
    name5 = "Accuracy of MSE using valid data per epoch"
    name6 = "Accuracy of MSE using test data per epoch"


    alpha = 0.001
    lam = 0
    train2, valid2, test2, atrain2, avalid2, atest2, iter_plt = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    alpha = 0.0001
    lam = 0
    train3, valid3, test3, atrain3, avalid3, atest3, iter_plt = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    y1 = [(train1, 'alpha = 0.005', 'r-'), \
         (train2, 'alpha = 0.001', 'g-'), \
         (train3, 'alpha = 0.0001', 'b-')]
    y2 = [(valid1, 'alpha = 0.005', 'r-'), \
         (valid2, 'alpha = 0.001', 'g-'), \
         (valid3, 'alpha = 0.0001', 'b-')]
    y3 = [(test1, 'alpha = 0.005', 'r-'), \
         (test2, 'alpha = 0.001', 'g-'), \
         (test3, 'alpha = 0.0001', 'b-')]

    y4 = [(atrain1, 'alpha = 0.005', 'r-'), \
         (atrain2, 'alpha = 0.001', 'g-'), \
         (atrain3, 'alpha = 0.0001', 'b-')]
    y5 = [(avalid1, 'alpha = 0.005', 'r-'), \
         (avalid2, 'alpha = 0.001', 'g-'), \
         (avalid3, 'alpha = 0.0001', 'b-')]
    y6 = [(atest1, 'alpha = 0.005', 'r-'), \
         (atest2, 'alpha = 0.001', 'g-'), \
         (atest3, 'alpha = 0.0001', 'b-')]


    plotFigures(name1, "Error", "Epoch", iter_plt, y1, 2)
    plotFigures(name2, "Error", "Epoch", iter_plt, y2, 2)
    plotFigures(name3, "Error", "Epoch", iter_plt, y3, 2)
    plotFigures(name4, "Accuracy", "Epoch", iter_plt, y4, 4)
    plotFigures(name5, "Accuracy", "Epoch", iter_plt, y5, 5)
    plotFigures(name6, "Accuracy", "Epoch", iter_plt, y6, 6)

    #plotFigures(name2, "epoch", "accuracy", iter_plt, y2, 4)'''

    #####
    #1.4#
    #####
    # alpha = 0.005
    # lam = 0.001
    # W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    # alpha = 0.005
    # lam = 0.1
    # W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    # alpha = 0.005
    # lam = 0.5
    # W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")


    #####
    #1.4 - with graphing#
    #####
    '''
    alpha = 0.005
    lam = 0.001
    train1, valid1, test1, atrain1, avalid1, atest1, iter_plt1  = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    alpha = 0.005
    lam = 0.1

    train2, valid2, test2, atrain2, avalid2, atest2, iter_plt2 = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    alpha = 0.005
    lam = 0.5
    train3, valid3, test3, atrain3, avalid3, atest3, iter_plt3  = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    name1 = "Error of training data using MSE per epoch"
    name2 = "Error of valid data using MSE per epoch"
    name3 = "Error of test data using MSE per epoch"
    y1 = [(train1, 'lambda = 0.001', 'r-'), \
          (train2, 'lambda = 0.1', 'g-'), \
          (train3, 'lambda = 0.5', 'b-')]
    y2 = [(valid1, 'lambda = 0.001', 'r-'), \
         (valid2, 'lambda = 0.1', 'g-'), \
         (valid3, 'lambda = 0.5', 'b-')]
    y3 = [(test1, 'lambda = 0.001', 'r-'), \
         (test2, 'lambda = 0.1', 'g-'), \
         (test3, 'lambda = 0.5', 'b-')]
    print(len(train1))

    plotFigures(name1, "Error", "Epoch", iter_plt3, y1, 2)
    plotFigures(name2, "Error", "Epoch", iter_plt3, y2, 2)
    plotFigures(name3, "Error", "Epoch", iter_plt3, y3, 2)


    #####
    #1.5#
    #####
    #Edit return values before USING
    starttime = os.times()[0]
    W = LeastSquares(trainData, trainTarget)
    error1 = MSE(W, 0, trainData, trainTarget, 0)
    endtime = os.times()[0]
    LStime = endtime - starttime

    starttime = os.times()[0]
    W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, 0.005, iterations, 0, 1*10**(-7), lossType="MSE")
    error2 = MSE(W, b, trainData, trainTarget, 0)
    endtime = os.times()[0]
    GDtime = endtime - starttime

    print(LStime)
    print(GDtime)
    print(error1)
    print(error2)

    '''

    #####
    #2.2#
    #####
    '''    alpha = 0.005
    lam = 0.1
    train1, valid1, test1, atrain1, avalid1, atest1, iter_plt = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
        testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="CE")

    name1 = "Error of training data using CE per epoch"
    name2 = "Accuracy of CE using training data per epoch"


    y1 = [(train1, 'Training', 'r-'), \
         (valid1, 'Valid', 'g-'), \
         (test1, 'Test', 'b-')]
    y2 = [(atrain1, 'Training', 'r-'), \
         (avalid1, 'Valid', 'g-'), \
         (atest, 'Test', 'b-')]


    plotFigures(name1, "Error", "Epoch", iter_plt, y1, 2)
    plotFigures(name2, "Error", "Epoch", iter_plt, y2, 2)
    '''
    #####
    #2.3#
    #####
    SGD(500, 700, "MSE")
    #SGD(500, 700, "CE")
    # alpha = 0.005
    # #Logistic Regression
    # train2, valid2, test2, atrain2, avalid2, atest2, iter_plt2 = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")
    # train2, valid2, test2, atrain2, avalid2, atest2, iter_plt2 = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="CE")
    #
    # name1 = "Logistic Regression"
    # name2 = "Cross Entropy Loss"
    #
    # y1 = [(train1, 'lambda = 0.001', 'r-'), \
    #       (train2, 'lambda = 0.1', 'g-'), \
    #       (train3, 'lambda = 0.5', 'b-')]
    # plotFigures(name1, "Error", "Epoch", iter_plt1, y1, 2)
    plt.show()


if __name__ == "__main__":
    main()
