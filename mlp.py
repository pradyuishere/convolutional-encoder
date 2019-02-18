import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def mlp(col_mat_in, num_layers, layer_sizes, weights, biases, activation_funcs):
    prod_out = col_mat_in
    for iter in range(num_layers):
#         this_wt = weights[iter]
        prod_out = activation_funcs[iter](np.matmul(np.transpose(weights[iter]), prod_out)+biases[iter])
    return prod_out
