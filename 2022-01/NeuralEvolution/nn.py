#homemake NeralNet class for practice

import numpy as np
import random


def randomize(x, e):
    #print(x)
    new = x.copy()
    for i in range(len(x)):
        for j in range(len(x[i])):
            if random.uniform(0, 1) < 0.10 - 0.00005*e:
                new[i][j] = random.uniform(-5, 5)
    return new

def activation(x):   
    return 1/(1+np.exp(-x))

def activationPrime(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

class NeuralNetwork:

    def __init__(self, input, hidden, output):
        self.dim = (input, hidden, output)

        self.w1 = np.array([[random.uniform(-1, 1) for i in range(input)] for j in range(hidden)])
        self.w2 = np.array([[random.uniform(-1, 1) for i in range(hidden)] for j in range(output)])
        self.b1 = np.array([[random.uniform(-1, 1)] for j in range(hidden)])
        self.b2 = np.array([[random.uniform(-1, 1)] for i in range(output)])

        self.activationf = lambda x: x if x>0 else 0

    def describe(self):
        print(self.w1, self.w2, self.b1, self.b2)

    def calculateGrad(self, x, y):
        a1 = np.dot(self.w1, x) + self.b1
        z1 = activation(a1)
        a2 = np.dot(self.w2, z1) + self.b2
        z2 = activation(a2)

        b2grad = 2 * (z2 - y) * activationPrime(a2)
        w2grad = z1.transpose() * activationPrime(a2) * 2 * (z2 - y)
        a1gradmat = self.w2.transpose() * (z2.transpose() - y.transpose()) * 2 * activationPrime(a2.transpose()) * activationPrime(a1)
        b1grad = np.array([a1gradmat.sum(axis = 1)]).transpose()
        w1grad = b1grad * x.transpose()

        return w1grad, w2grad, b1grad, b2grad
    
    def backpropagate(self, x, y, lr = 0.1):
        w1grad = np.array([[0. for i in range(self.dim[0])] for j in range(self.dim[1])])
        w2grad = np.array([[0. for i in range(self.dim[1])] for j in range(self.dim[2])])
        b1grad = np.array([[0.] for j in range(self.dim[1])])
        b2grad = np.array([[0.] for i in range(self.dim[2])])

        for i in range(len(x)):
            grad = self.calculateGrad(x[i], y[i])
            w1grad += grad[0]; w2grad += grad[1]; b1grad += grad[2]; b2grad += grad[3]

        self.w1 -= lr * w1grad / len(x)
        self.w2 -= lr * w2grad / len(x)
        
        self.b2 -= lr * b2grad / len(x)
        self.b1 -= lr * b1grad / len(x)

    def predict(self, x):
        temp1 = np.dot(self.w1, x) + self.b1
        h = activation(temp1)
        return activation(np.dot(self.w2, h)+self.b2)

    def copy(self):
        new = NeuralNetwork(self.dim[0], self.dim[1], self.dim[2])
        new.w1 = self.w1.copy()
        new.w2 = self.w2.copy()
        new.b1 = self.b1.copy()
        new.b2 = self.b2.copy()

        return new

    def cost(self, x, y):
        
        s = 0
        for i in range(len(x)):
            pred = self.predict(x[i])
            squaredError = (pred - y[i]) ** 2
            s += sum(squaredError)
        return s/len(x)
            

    def mutate(self, e):
        neww1 = randomize(self.w1, e)
        neww2 = randomize(self.w2, e)
        newb1 = randomize(self.b1, e)
        newb2 = randomize(self.b2, e)

        self.w1 = neww1
        self.w2 = neww2
        self.b1 = newb1
        self.b2 = newb2

if __name__ == '__main__':
    x = np.array([[[0], [1], [2], [3]], [[2], [3], [4], [5]]])
    y = np.array([[[0], [1]], [[1], [0]]])
    test = NeuralNetwork(4, 3, 2)
    print(test.cost(x, y))
    for i in range(1000):
        test.backpropagate(x, y)
    print(test.predict(x[0]))