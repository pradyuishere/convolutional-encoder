import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def conv_layer(input_img, num_kernels, *kernels, stride = (1, 1), nonlinear_func, pad = 'same'):
    if pad =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+kernels[0].shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+kernels[0].shape[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((img.shape[0], img.shape[1], num_kernels))
    else:
        dimy = input_img.shape[0]
        dimx = input_img.shape[1]
        img_padded = input_img
        img_out = np.zeros((int((dimy-kernels[0].shape[0])/stride[0])+1, (int(dimx-kernels[0].shape[1])/stride[1])+1, num_kernels))

    for iter in range(num_kernels):
        img_out[:, :, iter] = conv2d(input_img, kernels[iter], nonlinear_func, stride, pad)

    return img_out
