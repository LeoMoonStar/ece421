import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

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
    while i < iterations or abs(error1 - error2) <= EPS:
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

    y1 = [(error_train_plt, 'training', 'r-'), \
         (error_valid_plt, 'valid', 'g-'), \
         (error_test_plt, 'test', 'b-')]
    y2 = [(acc_train_plt, 'training', 'r-'), \
         (acc_valid_plt, 'valid', 'g-'), \
         (acc_test_plt, 'test', 'b-')]
    name1 = "Training error of " + lossType + " per epoch"
    name2 = "accuracy of " + lossType + " per epoch"

    plotFigures(name1, "epoch", "error", iter_plt, y1, 2)
    plotFigures(name2, "epoch", "accuracy", iter_plt, y2, 4)
    return W, b

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    dim_x = trainData.shape[1]
    dim_y = trainData.shape[2]

    tf.set_random_seed(421)
    W_shape = (dim_x*dim_y, 1)
    W = tf.get_variable("W", initializer=tf.truncated_normal(shape=W_shape, stddev=0.5))
    b = tf.get_variable("b", initializer=tf.truncated_normal(shape=[1], stddev=0.5))

    X = tf.placeholder(tf.float32, shape=(500, dim_x*dim_y), name="X")
    Y = tf.placeholder(tf.float32, shape=(500, 1), name="Y")
    lam = tf.placeholder(tf.float32, shape=(1, None), name="lam")

    predict = None
    loss = None
    if lossType == "MSE":
        predict = tf.matmul(X, W) + b
        loss = tf.losses.mean_squared_error(labels=Y, predictions=predict)
    elif lossType == "CE":
        logit = -1*(tf.matmul(X, W) + b)
        predict = tf.sigmoid(logit)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logit)

    acc = tf.count_nonzero(tf.greater(predict, 0.5))/predict.shape[0]
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)

    return W, b, predict, Y, X, loss, train_op, acc, lam

def SGDBatchs(it, X, Y, batchSize, sess, train_op, loss, acc):
    data_shape = (batchSize, X.shape[1]*X.shape[2])
    target_shape = (batchSize, 1)
    losses = []
    acc = []

    for i in range(it):
        x_batch = X[i*batchSize:(i+1)*batchSize]
        y_batch = Y[i*batchSize:(i+1)*batchSize]
        _, l, a = sess.run([train_op, loss, acc], feed_dict={X:x_batch, Y:y_batch})
        losses.append(l)
        acc.append(a)

    return losses, acc

def SGD(batchSize, iterations, type):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    W, b, predict, Y, X, loss, train_op, acc, lam = buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-08, lossType=type, learning_rate=0.001)

    data_shape = (batchSize, trainData.shape[1]*trainData.shape[2])
    target_shape = (batchSize, 1)

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
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(iterations):
            np.random.shuffle(trainData)
            np.random.shuffle(validData)
            np.random.shuffle(testData)

            l_t, a_t = SGDBatchs(trainBatches, trainData, trainTarget, \
                batchSize, sess, train_op, loss, acc)
            l_v, a_v = SGDBatchs(validBatches, validTarget, trainTarget, \
                batchSize, sess, train_op, loss, acc)
            l_t, a_t = SGDBatchs(testBatches, testTarget, trainTarget, \
                batchSize, sess, train_op, loss, acc)

    iter_plt1 = range(iterations*)
    y1 = [(l_train, 'training', 'r-'), \
         (l_valid, 'valid', 'g-'), \
         (l_test, 'test', 'b-')]
    y2 = [(a_train, 'training', 'r-'), \
         (a_valid, 'valid', 'g-'), \
         (a_test, 'test', 'b-')]
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
    # alpha = 0.005
    # lam = 0
    # W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    # alpha = 0.001
    # lam = 0
    # W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

    # alpha = 0.0001
    # lam = 0
    # W, b = grad_descent(weights, bias, trainData, trainTarget, validData, validTarget, \
    #     testData, testTarget, alpha, iterations, lam, 1*10**(-7), lossType="MSE")

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
    #1.5#
    #####

    #####
    #2.2#
    #####

    #####
    #2.3#
    #####
    # SGD(500, 700, "MSE")
    plt.show()


if __name__ == "__main__":
    main()