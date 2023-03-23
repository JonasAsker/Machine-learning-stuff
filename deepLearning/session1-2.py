import numpy as np
import random
import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

m = 10000
k = 5000
alpha = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.001, 0.0001]

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def generateData():
    x1Train = []; x2Train = []; yTrain = [];
    xTrain = np.array([]);
    for i in range(m):
        x1Train.append(random.uniform(-10,10))
        x2Train.append(random.uniform(-10,10))

        if (x1Train[-1] < -5 or x1Train[-1] > 5): # change this line for session zero or one
            yTrain.append(1) # if in upper left triangle of xy plane we say 1 otherwise 0
        else:
            yTrain.append(0)

    x1Train = np.array(x1Train); x2Train = np.array(x2Train); yTrain = np.array(yTrain);
    xTrain = np.vstack([x1Train, x2Train]);
    return xTrain, yTrain;

def forwardPropagate(xTrain, W1, W2, b1, b2):
    z1 = np.matmul(W1.T, xTrain) + b1
    A1 = sigmoid(z1)
    z2 = np.matmul(W2.T, A1) + b2
    A2 = sigmoid(z2)
    return A1, A2, z1

def backPropagate(A1, A2, W2, Z1, xTrain, yTrain):
    dZ2 = A2 - yTrain
    dW2 = (1/float(m)) * np.matmul(dZ2, A1.T)
    db2 =  (1/float(m)) * np.sum(dZ2)
    dZ1 = np.matmul(W2, dZ2) * sigmoid(Z1) * (1 - sigmoid(Z1))
    dW1 = (1/float(m)) * np.matmul(dZ1, xTrain.T)
    db1 = (1/float(m)) * np.sum(dZ1)
    return dW2, dW1, db2, db1

def propagate(xTrain, yTrain, W1, W2, b1, b2):
    A1, A2, Z1 = forwardPropagate(xTrain, W1, W2, b1, b2)
    dW2, dW1, db2, db1 = backPropagate(A1, A2, W2, Z1, xTrain, yTrain)
    return dW2, dW1, db2, db1

def predict(xTrain, yTrain, W1, W2, b1, b2):
    A = sigmoid(np.dot(W1.T, xTrain) + b1)
    APrime = sigmoid(np.dot(W2.T, A) + b2)
    yPrediction = APrime > 0.5
    return np.sum(yPrediction == yTrain) / m

def main():
    xTrain, yTrain = generateData()
    results = []
    for j in range(10):
        b1 = random.uniform(-1, 1); b2 = random.uniform(-1, 1);
        W1 = np.random.randn(2,2); W2 = np.random.randn(2,1);
        for i in range(k):
            dW2, dW1, db2, db1 = propagate(xTrain, yTrain, W1, W2, b1, b2)
            W1 -= alpha[j] * dW1
            W2 -= alpha[j] * dW2.T
            b1 -= alpha[j] * db1
            b2 -= alpha[j] * db2
        res = predict(xTrain, yTrain, W1, W2, b1, b2)
        results.append(res)
    print(len(alpha), len(results))
    plt.plot(alpha, results)
    plt.show()

main()