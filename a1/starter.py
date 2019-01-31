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
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))
    WX = np.matmul(X_aux, W_aux)
    cost = WX + b - y
    cost = np.sum(cost*cost)/(2*num_train_ex)

    #regularization
    regu = np.matmul(W_aux.transpose(), W_aux)*reg/2

    return np.sum(cost+regu)
    
def gradMSE(W, b, x, y, reg):
    # grad with respect weights
    num_train_ex = x.shape[0]
    num_pixels = x.shape[1]*x.shape[2]
    W_aux = W.reshape((num_pixels, 1))
    X_aux = x.reshape((num_train_ex, num_pixels))
    #print(W_aux.shape, X_aux.shape, np.matmul(X_aux, W_aux).shape, y.shape, (np.matmul(X_aux, W_aux) + b - y).shape)
    c = np.matmul(X_aux, W_aux) + b - y
    #print(c)
    #print(np.matmul(X_aux.transpose(), c))
    grad_W = np.matmul(X_aux.transpose(), c)/num_train_ex + reg*W_aux
    # temp = np.array([[1, 1, 1],[2, 2, 2],[3, 3, 3]])
    # temp2 = np.array([[1],[2],[3]])
    #print(temp.shape, temp2.shape, temp*temp2)
    # print(c.shape, (X_aux*c).shape, grad_W.shape)

    grad_b = np.sum(c)/num_train_ex

    return grad_W.reshape((x.shape[1], x.shape[2])), grad_b

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
        W_grad, b_grad = gradMSE(W, b, trainingData, trainingLabels, reg)
        W = W - alpha*W_grad
        b = b - alpha*b_grad
        error = MSE(W, b, trainingData, trainingLabels, reg)
        if i % 1000 == 0:
            print(error)
        #print(error)
        i += 1
    return W, b

def buildGraph(beta1=None, beta2=None, epsilon=None, lossType=None, learning_rate=None):
    # Your implementation here
    return

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
#W = np.array([[1,2],[3,4]])
#x = np.array([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]]])
#y = np.array([[1], [2], [1]])
#print(gradMSE(W, 1, x, y, 2))
W = np.random.rand(trainData.shape[1], trainData.shape[2])
alpha = 0.0001
grad_descent(W , 0, trainData, trainTarget, alpha, 5000, 0, 1*10**(-7))