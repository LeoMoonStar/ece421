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
    #print("start")
    #print(xn)
    #print(":)")
    #print((-1)*xn)
    #print(":)")
    #print(np.exp((-1)*xn))
    #print(":)")
    #print(1+np.exp((-1)*xn))
    #print(":)")

    sigma = 1 / (1 + np.exp((-1)*xn))
    #print(sigma)
    #print(":)")

    #print(1-sigma)

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

def grad_descent(W, b, trainingData, trainingLabels, alpha, iterations, reg, EPS, lossType="None"):
    i = 0
    iter_plt = []
    error_plt = []

    error1 = float("inf")
    error2 = 0   
    b_grad = None
    W_grad = None
    while i < iterations or abs(error1 - error2) <= EPS:
        W = W - alpha*W_grad
        b = b - alpha*b_grad
        error1 = error2

        if lossType == "MSE":
            error2 = MSE(W, b, trainingData, trainingLabels, reg)
        elif lossType == "CE":
            error2 = crossEntropyLoss(W, b, trainingData, trainingLabels, reg)

        iter_plt.append(i)
        error_plt.append(error2)
        if i % 1000 == 0:
            print(error2)

        #print(error)
        i += 1
    return W, b, error_plt, iter_plt

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

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)

    return W, b, predict, Y, X, loss, train_op, lam

def SGD(batchSize, iterations):
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

    batches = trainData.shape[0]/batchSize
    W, b, predict, Y, X, loss, train_op, lam = buildGraph(beta1=0.9, beta2=0.999, epsilon=1e-08, lossType="CE", learning_rate=0.001)
    losses = []


    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        for i in range(iterations):
            rand_i = np.random.choice(100, size=batchSize)

            x_batch = trainData[rand_i].reshape((batchSize, trainData.shape[1]*trainData.shape[2]))
            y_batch = trainTarget[rand_i].reshape((batchSize, 1))
            _, c = sess.run([train_op, loss], feed_dict={X:x_batch, Y:y_batch})
    return


#plotting
trainingData, validData, testData, trainTarget, validTarget, testTarget = loadData()
W = np.zeros((trainData.shape[1], trainData.shape[2]))
b = 0
alpha = 0.001
lam = 0

plt.figure(1)
W, b, e_arr, i_arr = grad_descent(W, b, trainData, trainTarget, alpha, 5000, lam, 1*10**(-7), lossType="MSE")
plt.plot(iter_plt, error_plt, 'g-')
e_arr, i_arr = grad_descent(W, b, validData, validTarget, alpha, 5000, lam, 1*10**(-7), lossType="MSE")
plt.plot(iter_plt, error_plt, 'b-')
e_arr, i_arr = grad_descent(W, b, testData, testTarget, alpha, 5000, lam, 1*10**(-7), lossType="MSE")
plt.plot(iter_plt, error_plt, 'r-')
plt.show()

#trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
#W = np.array([[1,2],[3,4]])
#x = np.array([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]]])
#y = np.array([[1], [0], [1]])
#print(LeastSquares(trainData,trainTarget))

#print(trainTarget)

#buildGraph(lossType = "MSE", learning_rate = 0.001)
#print(crossEntropyLoss(W, 1, x, y, 1))
#W = np.random.rand(trainData.shape[1], trainData.shape[2])
#W = np.zeros((trainData.shape[1], trainData.shape[2]))
#alpha = 0.001
#lam = 0.1
#W, b = grad_descent(W , 0, trainData, trainTarget, alpha, 5000, lam, 1*10**(-7), lossType="CE")
#print(W, b.shape)
#W, b = grad_descent(W, b, validData, validTarget, alpha, 5000, 0, 1*10**(-7))
#grad_descent(W, b, testData, testTarget, alpha, 5000, 0, 1*10**(-7))

#print('TensorFlow version: {0}'.format(tf.__version__))
#SGD(500, 700)

