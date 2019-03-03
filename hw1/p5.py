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
    kernels.append(np.random.normal(size = [ker_dim_layer2, ker_dim_layer2, ker_dim_layer1]))
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

def ReLU(x):
    mask  = (x >0) * 1.0
    return mask * x

def der_ReLU(x):
    mask  = (x >0) * 1.0
    return mask

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum()

def pool_func(img):
    return img.max()

num_epoch = 1
num_img = 1
for iter in range(num_epoch):
    for iter2 in range(num_img):
        this_img_in = X_train[:, :, iter2]
        this_img_out = Y_train[iter2]

        img_conv_1 = (conv_layer(this_img_in, ker_nums_layer1, ReLU, kernels[0:ker_nums_layer1], biases[0:ker_nums_layer1], stride = (1, 1), pad = 'same'))
        img_pool_1 = pool_layer(img_conv_1, pool_func, (2, 2), (2, 2))

        img_conv_2 = (conv_layer(img_pool_1, ker_nums_layer2, ReLU, kernels[ker_nums_layer1:ker_nums_layer2], biases[ker_nums_layer1:ker_nums_layer2], stride = (1, 1), pad = 'same'))
        img_pool_2 = pool_layer(img_conv_2, pool_func, (2, 2), (2, 2))

        dense_1_in = np.transpose(np.matrix(img_pool_2.flatten()))

        dense_1_out = activation_funcs[0](np.matmul(np.transpose(weights[0], dense_1_in)+bias[0]))
        dense_2_out = activation_funcs[1](np.matmul(np.transpose(weights[1], dense_2_out)+bias[1]))

        final_out = softmax(dense_2_out)

        grad_2_wt = np.matmul((final_out-this_img_out), np.transpose(dense_1_out))
        grad_2_wt = np.transpose(grad_2_wt)
        grad_2_bi = (final_out-this_img_out)
        grad_2_pre = (final_out-this_img_out)

        grad_1_wt = np.matmul(np.multiply(np.matmul(weights[1], grad_2_pre), der_ReLU(dense_1_out)), np.transpose(dense_1_in))
        grad_1_wt = np.transpose(grad_1_wt)
        grad_1_bi = np.multiply(np.matmul(weights[1], grad_2_wt), der_ReLU(dense_1_out))
        grad_1_pre = np.multiply(np.matmul(weights[1], grad_2_pre), der_ReLU(dense_1_out))

        cc_out = np.matmul(weights[0], grad_1_pre)

        cc_err_out = np.reshape(cc_out, img_pool_2.shape)

        grad_conv_2 = []
        unpool_layer_2_mask = np.equal(img_conv_2, img_pool_2.repeat(2, axis = 0).repeat(2, axis = 1))
        cc_err_back_layer2 = np.multiply(unpool_layer_2_mask, cc_err_out.repeat(2, axis = 0).repeat(2, axis = 1))
        for iter in range(ker_nums_layer2):
            grads_1 = np.zeros([5, 5, ker_nums_layer1])
            for iter2 in range(ker_nums_layer1):
                grads_1[:, :, iter2] = (conv2d(img_pool_1[:, :, iter2], np.rot90(np.multiply(der_ReLU(img_conv_2[:, :, iter]), cc_err_out[:, :, iter]), 2), linear_func, (1,1), 'valid'))
            grad_conv_2.append(grads_1)

        biases_conv_2 = []
        for iter in range(ker_nums_layer2):
            biases_conv_2.append(np.multiply(der_ReLU(img_conv_2[:, :, iter]), cc_err_out[:, :, iter]))
        
