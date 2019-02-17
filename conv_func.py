import cv2
import numpy as np

def padding (img, dimx, dimy):
    dimx_zeros = dimx - img.shape[1]
    dimy_zeros = dimy - img.shape[0]
    dimx_left = int(dimx_zeros/2)
    dimx_right = dimx_zeros-dimx_left
    dimy_top = int(dimy_zeros/2)
    dimy_down = dimy_zeros-dimy_top

    img_out = np.zeros((dimy, dimx, img.shape[2]))

    img_out[dimy_top:dimy-dimy_bottom-1, dimx_left: dimx-dimx_right-1] = img

    return img_out


def conv2d (input_img, ker, nonlinear_func, stride=(1,1), padding='same'):
    if padding =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+ker.shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+ker.shape[1]
        img_padded = padding(img, dimx, dimy)
##
img = cv2.imread('img.jpg', 1)
img_out = padding(img, img.shape[1], img.shape[0])
cv2.imshow('image',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()
