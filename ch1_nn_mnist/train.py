#_*_coding:utf8_*_
import cv2
import numpy as np
import os

#手写数字训练代码，运行前需要先解压dataset.zip得到training_img和test_img
def get_training_data():
    images = []
    labels = []
    files = os.listdir("training_img") #获取训练数据列表
    for f in files:                  #读取每一张图片，并加入到样本中
        img = cv2.imread("training_img/" + f, cv2.IMREAD_GRAYSCALE) #以灰度图的格式读取
        img = img.astype(np.float32).reshape(1024) #默认读取的是8bit整数，转换为浮点数，并将32 * 32展开成1024维
        images.append(img)           #样本加入list中
        num = int(f[0])              #文件名的第一个字符就是标签，这里需要转成int型, 例如8_001.jpg是数字8
        label = np.zeros(10, dtype=np.float32)
        label[num] = 1               #标签是以one-hot方式存放的，举例[0,0,1,0,0,0,0,0,0,0]代表2
        labels.append(label)         #图片对应标签也加入到list
    return (np.array(images), np.array(labels)) #转换为numpy 的数组用于训练
 
net = cv2.ml.ANN_MLP_create()        #创建一个神经网络
net.setLayerSizes(np.array([1024, 16, 10]))  #1024个输入神经元，16个隐藏神经元, 10个输出神经元,
                                             #其中输入输出必须固定，隐藏神经元可调
net.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM) #激活函数设置为sigmoid
net.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)  #训练方法设置为反向传播
net.setBackpropWeightScale(0.1)              #学习速率，也就是权重更新的速率
net.setBackpropMomentumScale(0.01)           #避免陷入局部最优，可以在一个局部最低点时以一定概率冲出去，所以叫冲量
net.setTermCriteria((cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 0.01)) #设置停止条件，
                                                                      #100次迭代或权重更新值小于0.01

images, labels = get_training_data()         #获取训练数据
net.train(images, cv2.ml.ROW_SAMPLE, labels) #开始训练
net.save("mnist.xml")                        #训练完成，将结果保存成mnist.xml
