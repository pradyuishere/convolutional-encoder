import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
X_train = np.vstack([img.reshape((28, 28)) for img in mnist.train.images])
Y_train = mnist.train.labels
X_test  = np.vstack([img.reshape(28, 28) for img in mnist.test.images])
Y_test  = mnist.test.labels

del mnist
