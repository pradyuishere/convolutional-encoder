import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def conv_net(input_img, num_layers, ker_nums, kernels, strides, paddings, nonlinear_funcs, pool_funcs, pool_windows, pool_strides):
    img_out = input_img
    current_ker_count = 0
    for iter in range(num_layers):
        img_out = conv_layer(img_out, ker_nums[iter], nonlinear_funcs[iter], kernels[current_ker_count:current_ker_count+ker_nums[iter]], strides[iter], paddings[iter] )
        current_ker_count = current_ker_count + ker_nums[iter]
        img_out = pool_layer(img_out, pool_funcs[iter], pool_windows[iter], pool_strides[iter])
        # print(img_out.shape)
    return img_out


img = cv2.imread('image.png')
num_layers = 2

ker_nums_layer1 = 2
ker_nums_layer2 = 3
ker_nums = []

ker_nums.append(ker_nums_layer1)
ker_nums.append(ker_nums_layer2)

kernels = []

kl1n1 = np.ones((4, 4, 3))
kl1n2 = np.zeros((4, 4, 3))
kl1n2[2, 2, :] = 1

kernels.append(kl1n1)
kernels.append(kl1n2)

kl2n1 = np.random.randint(2, size = (6, 6, 2))
kl2n2 = np.ones((6, 6, 2))
kl2n3 = np.random.randint(10, size = (6, 6, 2))

kernels.append(kl2n1)
kernels.append(kl2n2)
kernels.append(kl2n3)

kernels = np.array(kernels)

# print(kernels[3].shape)

stridel1 = (2, 2)
stridel2 = (3, 3)

strides = []
strides.append(stridel1)
strides.append(stridel2)

# print(strides)

paddingl1 = 'same'
paddingl2 = 'valid'

paddings = []
paddings.append(paddingl1)
paddings.append(paddingl2)

# print(paddings)

nonlinear_funcl1 = nonlinear_func
nonlinear_funcl2 = nonlinear_func

nonlinear_funcs = []
nonlinear_funcs.append(nonlinear_funcl1)
nonlinear_funcs.append(nonlinear_funcl2)

# print(nonlinear_funcs)

pool_funcl1 = pool_func
pool_funcl2 = pool_func

pool_funcs = []
pool_funcs.append(pool_funcl1)
pool_funcs.append(pool_funcl2)

# print(pool_funcs)

pool_windowl1 = (3, 3)
pool_windowl2 = (2, 2)

pool_windows = []
pool_windows.append(pool_windowl1)
pool_windows.append(pool_windowl2)

# print(pool_windows)

pool_stridel1 = (2, 2)
pool_stridel2 = (1,1)

pool_strides = []
pool_strides.append(pool_stridel1)
pool_strides.append(pool_stridel2)

# print(pool_strides)

img_out3 = conv_net(img, num_layers, ker_nums, kernels, strides, paddings, nonlinear_funcs, pool_funcs, pool_windows, pool_strides)
