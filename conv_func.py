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
    #print(img.shape)
    return np.multiply(img, ker).sum()

def nonlinear_func(img):
    return img

def conv2d (input_img, ker, nonlinear_func, stride=(1,1), pad='same'):
    img_out = []
    if pad =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+ker.shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+ker.shape[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((img.shape[0], img.shape[1]))
    else:
        dimy = input_img.shape[0]
        dimx = input_img.shape[1]
        img_padded = input_img
        img_out = np.zeros((int((dimy-ker.shape[0])/stride[0])+1, (int((dimx-ker.shape[1])/stride[1])+1)))

    ker_rev = np.zeros([ker.shape[1], ker.shape[0], ker.shape[2]])
    for iter in range(ker.shape[0]):
        for iter2 in range(ker.shape[1]):
            ker_rev[ker.shape[1]-1-iter2, ker.shape[0]-1-iter] = ker[iter, iter2]

    ker_rev_y = ker_rev.shape[0]
    ker_rev_x = ker_rev.shape[1]
    print(ker_rev_y)
    print(ker_rev_x)
    print(img_padded.shape)
    for iter in range(int((dimy-ker_rev.shape[0])/stride[0]) +1):
        for iter2 in range(int((dimx-ker_rev.shape[1])/stride[1])+1 ):
	    #print(iter)
	    #print(iter2)
            img_out[iter, iter2] =corr2d(img_padded[iter*stride[0]:iter*stride[0]+ker_rev_y, iter2*stride[1]:iter2*stride[1]+ker_rev_x], ker_rev)
    print(img_out.shape)

    return nonlinear_func(img_out)

def pool_func(img):
    return img.min()

def pooling(input_img, pool_func, pool_window=(1,1), stride = (1,1)):
    dimx = input_img.shape[1]
    dimy = input_img.shape[0]
    img_out = np.zeros((int((dimy-pool_window[0])/stride[0])+1, (int((dimx-pool_window[1])/stride[1])+1)))

    pool_window_x = pool_window[1]
    pool_window_y = pool_window[0]

    for iter in range(int((dimy-pool_window_y)/stride[0]) +1):
        for iter2 in range(int((dimx-pool_window_x)/stride[1])+1 ):
	    #print(iter)
	    #print(iter2)
            img_out[iter, iter2] =pool_func(input_img[iter*stride[0]:iter*stride[0]+pool_window_y, iter2*stride[1]:iter2*stride[1]+pool_window_x])
    print(img_out.shape)
    return img_out
##
img = cv2.imread('image.png')
ker1 = np.ones((10, 10, 3))/300
print(img.shape)
img_out1 = conv2d(img, ker1, nonlinear_func, stride = (5, 5), pad = 'valid')
img_out1 = pooling(img_out1, pool_func, pool_window=(2,2), stride = (2,2))
#img_out1 = padding(img, img.shape[1]+10, img.shape[0]+1000)
plt.imshow(img_out1.astype(int), cmap = 'gray')
plt.show()
