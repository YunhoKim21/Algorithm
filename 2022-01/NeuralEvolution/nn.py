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

    def predict(self, x):
        data = np.array(x)
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
    test = NeuralNetwork(2, 2, 1)
    print('pred1:', test.predict([[1], [0]]))
    test.mutate()
    print('pred2:', test.predict([[1], [0]]))
    test.describe()