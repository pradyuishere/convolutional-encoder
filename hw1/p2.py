import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
X_train_temp = np.vstack([img.reshape((28, 28)) for img in mnist.train.images])
Y_train_temp = mnist.train.labels
X_test_temp  = np.vstack([img.reshape(28, 28) for img in mnist.test.images])
Y_test_temp  = mnist.test.labels

del mnist

X_train = np.zeros([28, 28, int(X_train_temp.shape[0]/28)])

for iter in range(int(X_train_temp.shape[0]/28)):
    X_train[:, :, iter] = X_train_temp[iter*28:(iter+1)*28, :]

X_test = np.zeros([28, 28, int(X_test_temp.shape[0]/28)])
for iter in range(int(X_test_temp.shape[0]/28)):
    X_test[:, :, iter] = X_test_temp[iter*28:(iter+1)*28, :]

Y_train = np.transpose(Y_train_temp)
Y_test = np.transpose(Y_test_temp)
