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
    print(img.dtype)

    return img_out.astype(int)


def conv2d (input_img, ker, nonlinear_func, stride=(1,1), padding='same'):
    if padding =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+ker.shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+ker.shape[1]
        img_padded = padding(img, dimx, dimy)
##
img = mpimg.imread('img.jpg')
img_out = padding(img, img.shape[1], img.shape[0])
plt.imshow(img_out)
plt.show()
