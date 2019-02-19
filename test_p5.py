import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def padding (img, dimx, dimy):
    dimx_zeros = dimx - img.shape[1]
    dimy_zeros = dimy - img.shape[0]
    dimx_left = int(dimx_zeros/2)
    dimx_right = dimx_zeros-dimx_left
    dimy_top = int(dimy_zeros/2)
    dimy_down = dimy_zeros-dimy_top

    img_out = np.zeros((dimy, dimx, img.shape[2]))
    img_out[dimy_top:dimy-dimy_down, dimx_left: dimx-dimx_right] = img

    return img_out



def corr2d (img, ker):
    return np.multiply(img, ker).sum()



def nonlinear_func(img):
    return img




def conv2d (input_img, ker, nonlinear_func, stride=(1,1), pad='same'):
    img_out = []
    if pad =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+ker.shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+ker.shape[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((input_img.shape[0], input_img.shape[1]))
    else:
        if((input_img.shape[0]-ker.shape[0])%stride[0]==0):
            dimy = input_img.shape[0]
        else:
            dimy = input_img.shape[0]+stride[0]-(input_img.shape[0]-ker.shape[0])%stride[0]
        if((input_img.shape[1]-ker.shape[1])%stride[1]==0):
            dimx = input_img.shape[1]
        else:
            dimx = input_img.shape[1]+stride[1]-(input_img.shape[1]-ker.shape[1])%stride[1]

        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((int((dimy-ker.shape[0])/stride[0])+1, (int((dimx-ker.shape[1])/stride[1])+1)))

    ker_rev = np.zeros([ker.shape[1], ker.shape[0], ker.shape[2]])

    for iter in range(ker.shape[0]):
        for iter2 in range(ker.shape[1]):
            ker_rev[ker.shape[1]-1-iter2, ker.shape[0]-1-iter] = ker[iter, iter2]

    ker_rev_y = ker_rev.shape[0]
    ker_rev_x = ker_rev.shape[1]
    for iter in range(int((dimy-ker_rev.shape[0])/stride[0]) +1):
        for iter2 in range(int((dimx-ker_rev.shape[1])/stride[1])+1 ):
            img_out[iter, iter2] =corr2d(img_padded[iter*stride[0]:iter*stride[0]+ker_rev_y, iter2*stride[1]:iter2*stride[1]+ker_rev_x], ker_rev)

    print("######################################################################")
    print("output size from the conv_layer : ", img_out.shape)
    print("ker size : ", ker.shape)
    print("stride : ", stride)
    print("pad : ", pad)
    return nonlinear_func(img_out)




def conv_layer(input_img, num_kernels, nonlinear_func, kernels, stride = (1, 1), pad = 'same'):
    if pad =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+kernels[0].shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+kernels[0].shape[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((input_img.shape[0], input_img.shape[1], num_kernels))
    else:
        if((input_img.shape[0]-kernels[0].shape[0])%stride[0]==0):
            dimy = input_img.shape[0]
        else:
            dimy = input_img.shape[0]+stride[0]-(input_img.shape[0]-kernels[0].shape[0])%stride[0]
        if((input_img.shape[1]-kernels[0].shape[1])%stride[1]==0):
            dimx = input_img.shape[1]
        else:
            dimx = input_img.shape[1]+stride[1]-(input_img.shape[1]-kernels[0].shape[1])%stride[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((int((dimy-kernels[0].shape[0])/stride[0])+1, (int((dimx-kernels[0].shape[1])/stride[1])+1), num_kernels))

    for iter in range(num_kernels):
        img_out[:, :, iter] = conv2d(input_img, kernels[iter], nonlinear_func, stride, pad)
    return img_out


def pool_func(img):
    return img.min()



def pooling(input_img, pool_func, pool_window=(1,1), stride = (1,1)):
    if((input_img.shape[1]-pool_window[1])%stride[1]==0):
        dimx = input_img.shape[1]
    else:
        dimx = input_img.shape[1]+stride[1]-(input_img.shape[1]-pool_window[1])%stride[1]

    if((input_img.shape[0]-pool_window[0])%stride[0]==0):
        dimy = input_img.shape[0]
    else:
        dimy = input_img.shape[0]+stride[0]-(input_img.shape[0]-pool_window[0])%stride[0]
    input_pad = np.zeros((input_img.shape[0], input_img.shape[1], 1))
    input_pad[:, :, 0] = input_img
    input_img = padding(input_pad, dimx, dimy)
    img_out = np.zeros((int((dimy-pool_window[0])/stride[0])+1, (int((dimx-pool_window[1])/stride[1])+1)))

    pool_window_x = pool_window[1]
    pool_window_y = pool_window[0]

    for iter in range(int((dimy-pool_window_y)/stride[0]) +1):
        for iter2 in range(int((dimx-pool_window_x)/stride[1])+1 ):
	    #print(iter)
	    #print(iter2)
            img_out[iter, iter2] =pool_func(input_img[iter*stride[0]:iter*stride[0]+pool_window_y, iter2*stride[1]:iter2*stride[1]+pool_window_x])
#     print(img_out.shape)
    print("######################################################################")
    print("output size from the pool_layer : ", img_out.shape)
    print("pool_window size : ", pool_window)
    print("stride : ", stride)
    return img_out


def pool_layer(input_img, pool_func, pool_window=(1,1), stride = (1,1)):
    if((input_img.shape[1]-pool_window[1])%stride[1]==0):
        dimx = input_img.shape[1]
    else:
        dimx = input_img.shape[1]+stride[1]-(input_img.shape[1]-pool_window[1])%stride[1]

    if((input_img.shape[0]-pool_window[0])%stride[0]==0):
        dimy = input_img.shape[0]
    else:
        dimy = input_img.shape[0]+stride[0]-(input_img.shape[0]-pool_window[0])%stride[0]

    input_img = padding(input_img, dimx, dimy)

    img_out = np.zeros((int((dimy-pool_window[0])/stride[0])+1, (int((dimx-pool_window[1])/stride[1])+1), input_img.shape[2]))
    for iter in range(input_img.shape[2]):
#         print(input_img.shape)
        img_out[:, :, iter] = pooling(input_img[:, :, iter], pool_func, pool_window, stride)
    return img_out


def conv_net(input_img, num_layers, ker_nums, kernels, strides, paddings, nonlinear_funcs, pool_funcs, pool_windows, pool_strides):
    img_out = input_img
    current_ker_count = 0
    images = []
    for iter in range(num_layers):
        img_out = conv_layer(img_out, ker_nums[iter], nonlinear_funcs[iter], kernels[current_ker_count:current_ker_count+ker_nums[iter]], strides[iter], paddings[iter] )
        current_ker_count = current_ker_count + ker_nums[iter]
        print(img_out.shape)
        img_out = pool_layer(img_out, pool_funcs[iter], pool_windows[iter], pool_strides[iter])
        print(img_out.shape)
        images.append(img_out)
    return images

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

kernels.append(kl1n1/kl1n1.sum())
kernels.append(kl1n2/kl1n2.sum())

kl2n1 = np.random.randint(2, size = (6, 6, 2))
kl2n2 = np.ones((6, 6, 2))
kl2n3 = np.random.randint(10, size = (6, 6, 2))

kernels.append(kl2n1/kl2n1.sum())
kernels.append(kl2n2/kl2n2.sum())
kernels.append(kl2n3/kl2n3.sum())

kernels = np.array(kernels)

# print(kernels[3].shape)

stridel1 = (2, 2)
stridel2 = (3, 3)

strides = []
strides.append(stridel1)
strides.append(stridel2)

# print(strides)

paddingl1 = 'valid'
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
pool_stridel2 = (2, 2)

pool_strides = []
pool_strides.append(pool_stridel1)
pool_strides.append(pool_stridel2)

# print(pool_strides)

img_out3 = conv_net(img, num_layers, ker_nums, kernels, strides, paddings, nonlinear_funcs, pool_funcs, pool_windows, pool_strides)

count = 1
for iter in range(np.array(img_out3).shape[0]):
    for iter2 in range(img_out3[iter].shape[2]):
        count+=1

plt.subplot(count, 1, 1)
plt.imshow(img)
plt.title("input")
current_count = 1
for iter in range(np.array(img_out3).shape[0]):
    for iter2 in range(img_out3[iter].shape[2]):
        current_count+=1
        plt.subplot(count, 1, current_count)
        plt.imshow(img_out3[iter][:, :, iter2], cmap = 'gray')
        plt.title("output")
plt.show()
