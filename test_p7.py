import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def nonlinear_func(img):
    return img

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()

def unravel(input_img, weight):
    input_row = np.matrix(input_img.flatten())
    input_col = np.transpose(input_row)
#     print(weight.shape)
    prod = np.matmul(np.transpose(weight), input_col)
    return prod

def mlp(col_mat_in, num_layers, layer_sizes, weights, biases, activation_funcs):
    prod_out = col_mat_in
    for iter in range(num_layers):
#         this_wt = weights[iter]
        prod_out = activation_funcs[iter](np.matmul(np.transpose(weights[iter]), prod_out)+biases[iter])
    return [softmax(prod_out), prod_out]

img = cv2.imread('image.png', 0)
temp = unravel(img, np.random.rand(img.size, 1024))
num_layers = 2
biases = []
weights = []

w1_mlp =  np.random.rand(temp.size, 512)
w2_mlp =  np.random.rand(512, 10)

b1_mlp = np.transpose(np.matrix(np.random.rand(512)))
b2_mlp = np.transpose(np.matrix(np.random.rand(10)))

layer_sizes = []
layer_sizes.append(512)
layer_sizes.append(10)

biases.append(b1_mlp)
biases.append(b2_mlp)

weights.append(w1_mlp)
weights.append(w2_mlp)

activation_funcs = []
activation_funcs.append(nonlinear_func)
activation_funcs.append(nonlinear_func)

# print(weights[0].dtype)

output_mlp = mlp(temp, num_layers, layer_sizes, weights, biases, activation_funcs)
print("output with softmax : ",output_mlp[0])
print("output without softmax : ", output_mlp[1])
