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
feature_map1_pooling = cnn.pooling(feature_map1_relu,2,2)

print('the first cnn end')

l2_filter = numpy.random.rand(3,5,5,feature_map1_pooling.shape[-1])
feature_map2_conv  = cnn.conv(feature_map1_pooling, l2_filter)
feature_map2_relu = cnn.relu(feature_map2_conv)
feature_map2_pooling = cnn.pooling(feature_map2_relu,2,2)
print('the second cnn end')

l3_filter = numpy.random.rand(5,7,7,feature_map2_pooling.shape[-1])
feature_map3_conv  = cnn.conv(feature_map2_pooling, l3_filter)
feature_map3_relu = cnn.relu(feature_map3_conv)
feature_map3_pooling = cnn.pooling(feature_map3_relu,2,2)
print('the third cnn end')

# 显示图片
# 原图片
fig0, ax0 = matplotlib.pyplot.subplots(nrows=1, ncols=1)
ax0.imshow(img).set_cmap("gray")
ax0.set_title("Input Image")
ax0.get_xaxis().set_ticks([])
ax0.get_yaxis().set_ticks([])
matplotlib.pyplot.savefig("Input_Image.png", bbox_inches="tight")
matplotlib.pyplot.close(fig0)

#第一层卷积结果图片
fig1, ax1 = matplotlib.pyplot.subplots(nrows=3, ncols=2)
ax1[0, 0].imshow(feature_map1_conv[:, :, 0]).set_cmap("gray")
ax1[0, 0].set_title("CNN1 Image")
ax1[0, 0].get_xaxis().set_ticks([])
ax1[0, 0].get_yaxis().set_ticks([])

ax1[0, 1].imshow(feature_map1_conv[:, :, 1]).set_cmap("gray")
ax1[0, 1].set_title("CNN2 Image")
ax1[0, 1].get_xaxis().set_ticks([])
ax1[0, 1].get_yaxis().set_ticks([])

ax1[1, 0].imshow(feature_map1_relu[:, :, 0]).set_cmap("gray")
ax1[1, 0].set_title("relu1 Image")
ax1[1, 0].get_xaxis().set_ticks([])
ax1[1, 0].get_yaxis().set_ticks([])

ax1[1, 1].imshow(feature_map1_relu[:, :, 1]).set_cmap("gray")
ax1[1, 1].set_title("relu2 Image")
ax1[1, 1].get_xaxis().set_ticks([])
ax1[1, 1].get_yaxis().set_ticks([])

ax1[2, 0].imshow(feature_map1_pooling[:, :, 0]).set_cmap("gray")
ax1[2, 0].set_title("pooling1 Image")
ax1[2, 0].get_xaxis().set_ticks([])
ax1[2, 0].get_yaxis().set_ticks([])

ax1[2, 1].imshow(feature_map1_pooling[:, :, 1]).set_cmap("gray")
ax1[2, 1].set_title("pooling2 Image")
ax1[2, 1].get_xaxis().set_ticks([])
ax1[2, 1].get_yaxis().set_ticks([])

matplotlib.pyplot.savefig("Lay1.png", bbox_inches="tight")
matplotlib.pyplot.close(fig1)

#第二层卷积结果图片
fig2, ax2 = matplotlib.pyplot.subplots(nrows=3, ncols=3)
ax2[0, 0].imshow(feature_map2_conv[:, :, 0]).set_cmap("gray")
ax2[0, 0].set_title("CNN1 Image")
ax2[0, 0].get_xaxis().set_ticks([])
ax2[0, 0].get_yaxis().set_ticks([])

ax2[0, 1].imshow(feature_map2_conv[:, :, 1]).set_cmap("gray")
ax2[0, 1].set_title("CNN2 Image")
ax2[0, 1].get_xaxis().set_ticks([])
ax2[0, 1].get_yaxis().set_ticks([])

ax2[0, 2].imshow(feature_map2_conv[:, :, 2]).set_cmap("gray")
ax2[0, 2].set_title("CNN3 Image")
ax2[0, 2].get_xaxis().set_ticks([])
ax2[0, 2].get_yaxis().set_ticks([])

ax2[1, 0].imshow(feature_map2_relu[:, :, 0]).set_cmap("gray")
ax2[1, 0].set_title("relu1 Image")
ax2[1, 0].get_xaxis().set_ticks([])
ax2[1, 0].get_yaxis().set_ticks([])

ax2[1, 1].imshow(feature_map2_relu[:, :, 1]).set_cmap("gray")
ax2[1, 1].set_title("relu2 Image")
ax2[1, 1].get_xaxis().set_ticks([])
ax2[1, 1].get_yaxis().set_ticks([])

ax2[1, 2].imshow(feature_map2_relu[:, :, 2]).set_cmap("gray")
ax2[1, 2].set_title("relu3 Image")
ax2[1, 2].get_xaxis().set_ticks([])
ax2[1, 2].get_yaxis().set_ticks([])

ax2[2, 0].imshow(feature_map2_pooling[:, :, 0]).set_cmap("gray")
ax2[2, 0].set_title("pooling1 Image")
ax2[2, 0].get_xaxis().set_ticks([])
ax2[2, 0].get_yaxis().set_ticks([])

ax2[2, 1].imshow(feature_map2_pooling[:, :, 1]).set_cmap("gray")
ax2[2, 1].set_title("pooling2 Image")
ax2[2, 1].get_xaxis().set_ticks([])
ax2[2, 1].get_yaxis().set_ticks([])

ax2[2, 2].imshow(feature_map2_pooling[:, :, 2]).set_cmap("gray")
ax2[2, 2].set_title("pooling3 Image")
ax2[2, 2].get_xaxis().set_ticks([])
ax2[2, 2].get_yaxis().set_ticks([])

matplotlib.pyplot.savefig("Lay2.png", bbox_inches="tight")
matplotlib.pyplot.close(fig2)

#Lay3 image
fig3, ax3 = matplotlib.pyplot.subplots(nrows=3, ncols=5)
ax3[0, 0].imshow(feature_map3_conv[:, :, 0]).set_cmap("gray")
ax3[0, 0].set_title("CNN1 Image")
ax3[0, 0].get_xaxis().set_ticks([])
ax3[0, 0].get_yaxis().set_ticks([])

ax3[0, 1].imshow(feature_map3_conv[:, :, 1]).set_cmap("gray")
ax3[0, 1].set_title("CNN2 Image")
ax3[0, 1].get_xaxis().set_ticks([])
ax3[0, 1].get_yaxis().set_ticks([])

ax3[0, 2].imshow(feature_map3_conv[:, :, 2]).set_cmap("gray")
ax3[0, 2].set_title("CNN3 Image")
ax3[0, 2].get_xaxis().set_ticks([])
ax3[0, 2].get_yaxis().set_ticks([])

ax3[0, 3].imshow(feature_map3_conv[:, :, 3]).set_cmap("gray")
ax3[0, 3].set_title("CNN4 Image")
ax3[0, 3].get_xaxis().set_ticks([])
ax3[0, 3].get_yaxis().set_ticks([])


ax3[0, 4].imshow(feature_map3_conv[:, :, 4]).set_cmap("gray")
ax3[0, 4].set_title("CNN5 Image")
ax3[0, 4].get_xaxis().set_ticks([])
ax3[0, 4].get_yaxis().set_ticks([])


ax3[1, 0].imshow(feature_map3_relu[:, :, 0]).set_cmap("gray")
ax3[1, 0].set_title("relu1 Image")
ax3[1, 0].get_xaxis().set_ticks([])
ax3[1, 0].get_yaxis().set_ticks([])

ax3[1, 1].imshow(feature_map3_relu[:, :, 1]).set_cmap("gray")
ax3[1, 1].set_title("relu2 Image")
ax3[1, 1].get_xaxis().set_ticks([])
ax3[1, 1].get_yaxis().set_ticks([])

ax3[1, 2].imshow(feature_map3_relu[:, :, 2]).set_cmap("gray")
ax3[1, 2].set_title("relu3 Image")
ax3[1, 2].get_xaxis().set_ticks([])
ax3[1, 2].get_yaxis().set_ticks([])

ax3[1, 3].imshow(feature_map3_relu[:, :, 3]).set_cmap("gray")
ax3[1, 3].set_title("relu4 Image")
ax3[1, 3].get_xaxis().set_ticks([])
ax3[1, 3].get_yaxis().set_ticks([])

ax3[1, 4].imshow(feature_map3_relu[:, :, 4]).set_cmap("gray")
ax3[1, 4].set_title("relu5 Image")
ax3[1, 4].get_xaxis().set_ticks([])
ax3[1, 4].get_yaxis().set_ticks([])


ax3[2, 0].imshow(feature_map3_pooling[:, :, 0]).set_cmap("gray")
ax3[2, 0].set_title("pool1 Image")
ax3[2, 0].get_xaxis().set_ticks([])
ax3[2, 0].get_yaxis().set_ticks([])

ax3[2, 1].imshow(feature_map3_pooling[:, :, 1]).set_cmap("gray")
ax3[2, 1].set_title("pool2 Image")
ax3[2, 1].get_xaxis().set_ticks([])
ax3[2, 1].get_yaxis().set_ticks([])

ax3[2, 2].imshow(feature_map3_pooling[:, :, 2]).set_cmap("gray")
ax3[2, 2].set_title("pool3 Image")
ax3[2, 2].get_xaxis().set_ticks([])
ax3[2, 2].get_yaxis().set_ticks([])

ax3[2, 3].imshow(feature_map3_pooling[:, :, 3]).set_cmap("gray")
ax3[2, 3].set_title("pool4 Image")
ax3[2, 3].get_xaxis().set_ticks([])
ax3[2, 3].get_yaxis().set_ticks([])

ax3[2, 4].imshow(feature_map3_pooling[:, :, 4]).set_cmap("gray")
ax3[2, 4].set_title("pool5 Image")
ax3[2, 4].get_xaxis().set_ticks([])
ax3[2, 4].get_yaxis().set_ticks([])


matplotlib.pyplot.savefig("Lay3.png", bbox_inches="tight")
matplotlib.pyplot.close(fig3)