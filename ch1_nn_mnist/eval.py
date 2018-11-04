#_*_coding:utf8_*_
import cv2
import numpy as np
import os

#评估神经网络的精度
net = cv2.ml.ANN_MLP_create()
net = net.load("mnist.xml")   #加载权重
files = os.listdir("test_img")
tp = 0
for f in files:
    img = cv2.imread("test_img/" + f, cv2.IMREAD_GRAYSCALE)
    x = img.astype(np.float32).reshape(1, 1024)
    y, output = net.predict(x) #预测
    if int(f[0]) == int(y):    #如果预测准确，true positive增加
        tp += 1
    
print "accuracy=",float(tp) / float(len(files)) #精度=true_positive / 全部样本
