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

####Testing the pool_layer
img = cv2.imread('image.png')
img_out1 = pool_layer(img, pool_func, pool_window=(2,2), stride = (2,2))
print(img_out1.shape)

plt.subplot(img_out1.shape[2]+1,1,1)
plt.imshow(img)
plt.title("input")

for iter in range(img_out1.shape[2]):
    plt.subplot(img_out1.shape[2]+1, 1, iter+2)
    plt.imshow(img_out1[:, :, iter], cmap = 'gray')
    plt.title("output_img")

plt.show()
