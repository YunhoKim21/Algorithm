from nn import NeuralNetwork
import tensorflow as tf
import matplotlib.pyplot as plt
import time

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(len(x_train))