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
 #   for iter in range(img.shape[0]):
#        for iter2 in range(img.shape[1]):
#            img_out[dimy_top+iter, dimx_left+iter2, :] = img[iter, iter2, :]
  #   	     print(img[iter, iter2, :])
 #	     print(img_out[iter+dimy_top, iter2+dimx_left, :])
    return img_out

def corr2d (img, ker):
    return np.multiply(img, ker).sum()

def ker(img):
    return np.zeros(img.shape)

def conv2d (input_img, ker, nonlinear_func, stride=(1,1), padding='same'):
    img_out = []
    if padding =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+ker.shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+ker.shape[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros(input_img.shape)
    else:
        dimy = input_img.shape[0]
        dimx = input_img.shape[1]
        img_padded = input_img
        img_out = np.zeros((dimy-ker.shape[0])/stride[0]+1, (dimx-ker.shape[1])/stride[1]+1)

    ker_rev = np.transpose(zeros(ker.shape))
    for iter in range(ker.shape[0]):
        for iter2 in range(ker.shape[1]):
            ker_rev[ker.shape[1]-1-iter2, ker.shape[0]-1-iter] = ker[iter, iter2]

    ker_rev_y = ker_rev.shape[0]
    ker_rev_x = ker_rev.shape[1]
    for iter in range((dimy-ker_rev.shape[0])/stride[0] + 1):
        for iter2 in range((dimx-ker_rev.shape[1])/stride[1] + 1):
            img_out[iter, iter2] = corr2d(img[iter*stride[0]:iter*stride[0]+ker_rev_y, iter2*stride[1]:iter2*stride[1]+ker_rev_x], ker_rev)

    return ker(img_out)


##
img = mpimg.imread('image.png')
img_out = padding(img, img.shape[1]+10, img.shape[0]+1000)
plt.imshow(img_out)
plt.show()
