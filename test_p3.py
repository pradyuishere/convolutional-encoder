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



def sigmoid(x):
    return 1/(1+np.exp(-x))

def nonlinear_func(img):
    return sigmoid(img)




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

# ####Testing the conv_layer
img = cv2.imread('image.png')
ker1 = np.random.normal(size = (10, 10, 3))
ker12 = np.random.normal(size = (10, 10, 3))
# print(ker1)
ker2 = []
ker2.append(ker1)
ker2.append(ker12)
img_out1 = conv_layer(img, 2, nonlinear_func, np.array(ker2),  stride = (5, 5), pad = 'valid')

print(img_out1.shape)
print("kernel : ", ker2)
plt.subplot(img_out1.shape[2]+1,1,1)
plt.imshow(img)
plt.title("input")

for iter in range(img_out1.shape[2]):
    plt.subplot(img_out1.shape[2]+1, 1, iter+2)
    plt.imshow(img_out1[:, :, iter], cmap = 'gray')
    plt.title("output_img")

plt.show()
