# -*- coding: utf-8 -*-
# @Time    : 2018/11/27 0027 下午 3:39
# @Author  : oyj
# @Email   : 1272662747@qq.com
# @File    : CNNTestExample.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib
import skimage
import numpy
from skimage import data, color
import CNNComponents as cnn


img = skimage.data.chelsea()
plt.imshow(img)
plt.show()
img = skimage.color.rgb2gray(img)
plt.imshow(img)
plt.show()

l1_filter = numpy.zeros((2,3,3))
# 垂直检测
l1_filter[0, :, :] = numpy.array([[[-1, 0, 1],
                                   [-1, 0, 1],
                                   [-1, 0, 1]]])
# 水平检测
l1_filter[1, :, :] = numpy.array([[[1,   1,  1],
                                   [0,   0,  0],
                                   [-1, -1, -1]]])

feature_map1_conv  = cnn.conv(img, l1_filter)
feature_map1_relu = cnn.relu(feature_map1_conv)
feature_map1_pooling = cnn.pooling(feature_map1_relu)

plt.imshow(feature_map1_pooling)
plt.show()



