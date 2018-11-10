#_*_coding:utf8_*_
import cv2
import numpy as np
import os

#手写数字识别demo, 其中mnist.xml需要通过运行train.py得到
net = cv2.ml.ANN_MLP_create()
net = net.load("mnist.xml") #加载权重
files = os.listdir("test_img")
for f in files:
    img = cv2.imread("test_img/" + f, cv2.IMREAD_GRAYSCALE)
    x = img.astype(np.float32).reshape(1, 1024)
    y, output = net.predict(x) #通过神经网络预测手写数字
    title = str(int(y))        #y就是输出的结果，但是y是一个float类型的，我们需要转成字符串

    big_img = cv2.resize(img, (256, 256)) #原图是32 * 32太小，为方便展示，放大一下
    cv2.putText(big_img, title, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2) #在图像左上角将检测结果写进去
    cv2.imshow("window", big_img) #显示图片
    k = cv2.waitKey(0)
    if k == 27:
        break

