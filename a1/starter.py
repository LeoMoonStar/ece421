import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
#Temp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    #cost
    num_train_ex = X.shape[0]
    num_pixels = X.shape[1]*X.shape[2]
    W_aux = W.flatten()
    X_aux = X.reshape((num_train_ex, num_pixels))
    WX = np.matmul(X_aux, W_aux)
    cost = WX + b - y
    cost = np.sum(cost*cost)/(2*num_train_ex)

    #regularization
    reg = W_aux.dot(W_aux)*reg/2

    return cost+reg
    
def gradMSE(W, b, x, y, reg):
    # grad with respect weights
    num_train_ex = X.shape[0]
    num_pixels = X.shape[1]*X.shape[2]
    W_aux = W.flatten()
    X_aux = X.reshape((num_train_ex, num_pixels))
    c = np.matmul(X_aux, W_aux) + b - y
    grad_W = np.sum(c*X_aux, axis=1)/num_train_ex + reg*W_aux

    grad_b = np.sum(c)

    return grad_W.reshape((X.shape[1], X.shape[2])), grad_b

def crossEntropyLoss(W, b, x, y, reg):
    # Your implementation here
    return

def gradCE(W, b, x, y, reg):
    # Your implementation here
    return

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS):
    i = 0
    error = float("inf")
    while i  < iterations or error <= EPS:

        i += 1
    return

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return

loadData()