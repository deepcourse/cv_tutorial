#_*_coding:utf8_*_
import cv2
import numpy as np
import os

#手写数字识别demo, 其中mnist.xml需要通过训练得到
net = cv2.ml.ANN_MLP_create()
net = net.load("mnist.xml") #加载权重
files = os.listdir("test_img")
for f in files:
    img = cv2.imread("test_img/" + f, cv2.IMREAD_GRAYSCALE)
    x = img.astype(np.float32).reshape(1, 1024)
    y, output = net.predict(x) #通过神经网络预测手写数字
    print y
    cv2.imshow("image", img) #展示这张图片，并在控制台输出预测值
    k = cv2.waitKey(0)
    if k == 27:
        break

