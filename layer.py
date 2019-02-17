import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def conv_layer(input_img, num_kernels, nonlinear_func, kernels, stride = (1, 1), pad = 'same'):
    if pad =='same':
        dimy = stride[0]*(input_img.shape[0]-1)+kernels[0].shape[0]
        dimx = stride[1]*(input_img.shape[1]-1)+kernels[0].shape[1]
        img_padded = padding(input_img, dimx, dimy)
        img_out = np.zeros((img.shape[0], img.shape[1], num_kernels))
    else:
        dimy = input_img.shape[0]
        dimx = input_img.shape[1]
        img_padded = input_img
        img_out = np.zeros((int((dimy-kernels[0].shape[0])/stride[0])+1, (int((dimx-kernels[0].shape[1])/stride[1])+1), num_kernels))

    for iter in range(num_kernels):
        print(kernels.shape)
        img_out[:, :, iter] = conv2d(input_img, kernels[iter], nonlinear_func, stride, pad)

    return img_out
################################################################################
##Remember to give it and array of images, not a 3 dim input, array of 2d inputs
def pool_layer(input_img, pool_func, pool_window=(1,1), stride = (1,1)):
    dimy = input_img[0].shape[0]
    dimx = input_img[0].shape[1]
    img_out = np.zeros((int((dimy-pool_window[0])/stride[0])+1, (int((dimx-pool_window[1])/stride[1])+1), input_img.shape[0]))
    for iter in range(input_img.shape[0]):
        print(input_img[1].shape)
        img_out[:, :, iter] = pooling(input_img[iter], pool_func, pool_window, stride)
    return img_out
################################################################################
##Testing the conv layer

####Testing the conv_layer and pool layer
img = cv2.imread('image.png')
ker1 = np.zeros((10, 10, 3))
ker12 = np.ones((10, 10, 3))/30000
ker1[5, 5, :] = 1
# print(ker1)
ker2 = []
ker2.append(ker1)
ker2.append(ker12)
img_out1 = conv_layer(img, 2, nonlinear_func, np.array(ker2),  stride = (5, 5), pad = 'valid')
print(img_out1.shape)
img_out1 = pool_layer(img_out1, pool_func, pool_window=(2,2), stride = (2,2))
# img_out1 = pooling(img_out1, pool_func, pool_window=(2,2), stride = (2,2))
# img_out1 = conv2d(img, ker1*1000, nonlinear_func, stride = (5, 5), pad = 'valid')
# plt.imshow(img_out1, cmap = 'gray')
#img_out1 = padding(img, img.shape[1]+10, img.shape[0]+1000)
# print(img_out1.shape[2])

fig = plt.figure(figsize=(img_out1.shape[0], img_out1.shape[1]))  # width, height in inches

for i in range(img_out1.shape[2]):
    sub = fig.add_subplot(img_out1.shape[2], 1, i + 1)
    sub.imshow(img_out1[:,:, i], cmap = 'gray', norm=None)

plt.show()
