import numpy as np
import random

m = 10000
k = 5000

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

def forwardPropagate(xTrain, W, b):
    z = np.matmul(W.T, xTrain) + b
    a = sigmoid(z)
    return a

def backPropagate(A, xTrain, yTrain):
    dYHat = A - yTrain
    dA = dYHat * sigmoid(A) * (1 - sigmoid(A))
    dW = (1/float(m)) * np.matmul(xTrain, dA.T)
    db = (1/float(m)) * np.sum(dA)
    return dW, db

def propagate(xTrain, yTrain, W, b):
    A = forwardPropagate(xTrain, W, b)
    dW, db = backPropagate(A, xTrain, yTrain)
    return dW, db

def predict(xTrain, yTrain, W, b):
    A = sigmoid(np.dot(W.T, xTrain) + b)
    yPrediction = A > 0.5
    return np.sum(yPrediction == yTrain) / m

def main():
    xTrain, yTrain = generateData();
    b = random.uniform(-1, 1);
    W = np.random.randn(2,1);
    res = predict(xTrain, yTrain, W, b)
    print(res)
    for i in range(k):
        dW, db = propagate(xTrain, yTrain, W, b)
        W -= dW
        b -= db
    res = predict(xTrain, yTrain, W, b)
    print(res)
    

main()