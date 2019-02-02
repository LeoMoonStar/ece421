import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#Temp
#import matplotlib
#matplotlib.use("Agg")
#import matplotlib.pyplot as plt

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
        # temp = trainData.reshape((trainData.shape[0], trainData.shape[1]*trainData.shape[2]))
        # print(trainData.shape, temp.shape, trainTarget.shape)
        # print("training data", trainData[0][0][:28])
        # print("temp data", temp[0][:28])
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

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

    sigma = 1 / (1 + np.exp(-xn))

    cross_entropy = np.sum((-y*np.log(sigma) - (1 - y)*np.log(1-sigma)))
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

    e_in = -y*sigma + (1 - y)*sigma*np.exp(xn)
    grad_W = np.matmul(X_aux.transpose(), e_in)/num_train_ex + reg*W_aux
    print(grad_W)

    grad_b = np.sum(e_in)/num_train_ex

    return grad_W.reshape((x.shape[1], x.shape[2])), grad_b

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    i = 0
    error1 = float("inf")
    error2 = 0
    iter_plt = []
    error_plt = []
    while i  < iterations or abs(error1 - error2) <= EPS:
        error1 = MSE(W, b, trainingData, trainingLabels, reg)
        W_grad, b_grad = gradMSE(W, b, trainingData, trainingLabels, reg)
        W = W - alpha*W_grad
        b = b - alpha*b_grad
        error2 = MSE(W, b, trainingData, trainingLabels, reg)

        iter_plt.append(i)
        error_plt.append(error2)
        if i % 1000 == 0:
            print(error2)
        #print(error)
        i += 1
    plt.plot(iter_plt, error_plt, 'ro')
    plt.axis([0, len(iter_plt), 0, max(error_plt)])
    plt.show()
    return W, b

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
W = np.array([[1,2],[3,4]])
x = np.array([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]]])
y = np.array([[1], [2], [1]])
#print(gradCE(W, 1, x, y, 1))
W = np.random.rand(trainData.shape[1], trainData.shape[2])
alpha = 0.00001
W, b = grad_descent(W , 0, trainData, trainTarget, alpha, 5000, 0, 1*10**(-7))
W, b = grad_descent(W, b, validData, validTarget, alpha, 5000, 0, 1*10**(-7))
grad_descent(W, b, testData, testTarget, alpha, 5000, 0, 1*10**(-7))
