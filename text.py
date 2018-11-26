# -*- coding: utf-8 -*-
# @Time    : 2018/11/26 0026 下午 1:48
# @Author  : oyj
# @Email   : 1272662747@qq.com
# @File    : text.py
# @Software: PyCharm
import numpy
import skimage
import matplotlib.pyplot as plt
from skimage import data, io
print(numpy.random.rand(2, 2))
img = skimage.data.chelsea()
plt.imshow(img)
plt.axis('off')
plt.show()