import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def unravel(input_img, weight):
    input_row = np.matrix(input_img.flatten())
    input_col = np.transpose(input_row)
#     print(weight.shape)
    prod = np.matmul(np.transpose(weight), input_col)
    return prod

img = cv2.imread('image.png', 0)
temp = unravel(img, np.random.rand(img.size, 1024))
print(temp.shape)
