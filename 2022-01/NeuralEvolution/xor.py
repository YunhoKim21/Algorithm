from nn import NeuralNetwork
import numpy as np
epochs = 2000
population = 100
X = [[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]]
y = [0, 1, 1, 0]
pool = []

for i in range(population):
    pool.append([NeuralNetwork(2, 2, 1), 100])
t = NeuralNetwork(2, 2, 1)
t.w1 = np.array([[6, 6], [-5, -5]])
t.w2 = np.array([[-10, -10]])
t.b1 = np.array([[-10], [2]])
t.b2 = np.array([[5]])
#pool[0][0] = t
def abs(x):
    if x>0:
        return x
    return (-1)*x

def evaluate(network):
    cost = 0
    for i in range(4):
        pred = 1
        if network.predict(X[i])<0.5:
            pred = 0


        cost += abs(pred - y[i])
    return cost

for epoch in range(epochs):
    print(epoch)
    overall = 0
    for i in range(population):
        score = evaluate(pool[i][0])
        overall += score
        pool[i][1] = score
    print(overall)
    
    pool.sort(key = lambda x : x[1])
    #print(poo)
    for i in range(int(population/2)):
        pool[i+50][0] = pool[i][0].copy()
        pool[i+50][0].mutate(epoch)

pool[0][0].describe()
for i in range(4):
    print(pool[0][0].predict(X[i]))

print(evaluate(pool[0][0]))