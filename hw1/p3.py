import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from set_of_func import *
from tensorflow.examples.tutorials.mnist import input_data

def relu(img):
    x = np.zeros(img.shape)
    np.maximum(x, 0, img)
    return x

def avg(img):
    return img.sum()/img.shape()

################################################################################
##Load the data set
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
################################################################################
##Initialize the sizes and weights for the conv_net
num_layers = 2

ker_nums_layer1 = 32
ker_nums_layer2 = 64

ker_nums = []
ker_nums.append(ker_dim_layer1)
ker_nums.append(ker_dim_layer2)

kernels = []

ker_dim_layer1 = 5
ker_dim_layer2 = 5

biases = []

for iter in range(ker_nums_layer1):
    kernels.append(np.random.normal(size = [ker_dim_layer1, ker_dim_layer1]))
    biases.append(np.random.normal())

for iter in range(ker_nums_layer2):
    kernels.append(np.random.normal(size = [ker_dim_layer2, ker_dim_layer2]))
    biases.append(np.random.normal())

strides = []
strides.append((1, 1))
strides.append((1, 1))

paddings = []
paddings.append('valid')
paddings.append('valid')

non_linear_funcs = []
non_linear_funcs.append(relu)
non_linear_funcs.append(relu)

pool_funcs = []
pool_funcs.append(pool_func)
pool_funcs.append(pool_func)

pool_windows = []
pool_windows.append((2, 2))
pool_windows.append((2, 2))

pool_strides = []
pool_strides.append((2, 2))
pool_strides.append((2, 2))
################################################################################
##Unravel weights setup
weight_dense = np.random.normal(size = [3136, 1024])
bias_dense   = np.random.normal(size = [1024, 1])

num_layers = 2

layer_sizes = []
layer_sizes.append(1024)
layer_sizes.append(10)

weights = []
weights.append(weight_dense)
weights.append(np.random.normal(size = [1024, 10]))

biases = []
biases.append(bias_dense)
biases.append(np.random.normal(size = [10, 1]))

activation_funcs = []
activation_funcs.append(relu)
activation_funcs.append(linear_func)
################################################################################
##function to calculate the end_errors

def end_error(outputs, train_out):
    return train_out-outputs

################################################################################
##function to calculate the cost, cross_entropy

def cost_cross_entropy(outputs, train_out):
    costs = np.zeros(outputs.shape)
    for iter in range(costs.size):
        if(train_out[iter]==1):
            costs[iter] = -np.log(outputs[iter])
        else:
            costs[iter] = -np.log(1-outputs[iter])
    return costs.sum()

################################################################################



################################################################################
##Find the grads of weights and biases of the mlp, remember to calculate the inputs_mlp
def grads_one_epoch_weights_mlp_soft(end_error,
    weights,
    biases,
    activation_funcs,
    activation_funcs_der,
    inputs_mlp):

    weights_del = []
    for iter in range(weights.shape[0]-1):
        weights_del_temp = np.matmul(np.multiply(end_error, activation_funcs_der(weights[weights.shape[0]-1-iter]), inputs_mlp[weights.shape[0]-1-iter])
        end_error = np.matmul(weights[weights.shape[0]-1-iter], np.multiply(end_error, activation_funcs_der(weights[weights.shape[0]-1-iter])))
        weights_del.append(weights_del_temp)
    weights_del1 = []
    for iter in range(weights.shape[0]-1):
        weights_del1.append(weights_del[weights_del.shape[0]-1-iter])
    return np.array(weights_del1)

def grads_one_epoch_biases_mlp_soft(end_error,
    weights,
    biases,
    activation_funcs,
    activation_funcs_der,
    inputs_mlp):

    biases_del = []
    for iter in range(weights.shape[0]-1):
        weights_del_temp = np.matmul(np.multiply(end_error, activation_funcs_der(weights[weights.shape[0]-1-iter]), inputs_mlp[weights.shape[0]-1-iter]))
        biases_del_temp  = np.multiply(end_error, activation_funcs_der(weights[weights.shape[0]-1-iter]), inputs_mlp[weights.shape[0]-1-iter])
        end_error = np.matmul(weights[weights.shape[0]-1-iter], np.multiply(end_error, activation_funcs_der(weights[weights.shape[0]-1-iter])))
        biases_del.append(biases_del_temp)
    biases_del1 = []
    for iter in range(weights.shape[0]-1):
        biases_del1.append(weights_del[weights_del.shape[0]-1-iter])
    return np.array(biases_del1)
################################################################################
##Ravel function
def ravel(input, size):
    return np.reshape(input, size)
################################################################################
##Backprop conv_layer

def conv_back_one_epock_one_ker(error_img,
    input_img,
    bias,
    non_linear_func,
    non_linear_func_der):

    bias_del = conv2d(input_img, error_img, bias)
