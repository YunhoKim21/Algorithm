import numpy as np
import random 

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def activationPrime(x):
    return np.exp(-x) / (1 + np.exp(-x)) ** 2

def onecolumn(n):
    return np.array([[1 for i in range(n)]]).transpose()

class LogisticRegression:
    def __init__(self, n):
        self.n = n
        self.w = np.array([[0.0 for i in range(n)]]).transpose()
        self.b = np.array([random.uniform(-1, 1)])
        self.b = np.array([0.0])
    
    def predict(self, x):
        return sigmoid(np.dot(x, self.w) + self.b)

    def gradient(self, x, y):
        z = self.predict(x)
        a = np.dot(x, self.w) + self.b
        ret = ((-(y/z - (onecolumn(len(x)) - y) / (onecolumn(len(x))) - z))) * activationPrime(a)
        grad = (ret.transpose() * np.array(x).transpose()).mean(axis = 1)
        return np.array([grad]).transpose(), np.array([ret.transpose()[0].mean()])

    def learn(self, x, y, lr = 1, n = 1):
        for i in range(n):
            grad = self.gradient(x, y)
            self.w -= lr * grad[0]
            self.b -= lr * grad[1]

    def describe(self):
        print('w : ', self.w)
        print('b : ', self.b)

    def cost(self, x, y):
        pred = self.predict(x)
        costs = y * np.log(pred) + (np.array([[1 for i in range(len(x))]]).transpose() - y) * np.log(np.array([[1 for i in range(len(x))]]).transpose() - pred)
        return -costs.transpose()[0].mean()

if __name__ == '__main__':
    x_data = [[1, 2], [2, 3], [0, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [0], [1], [1]]
    model = LogisticRegression(2)
    print(model.cost(x_data, y_data))
    model.learn(x_data, y_data, n = 1000)
    print(model.cost(x_data, y_data))
    print(model.predict(x_data))
    model.describe()