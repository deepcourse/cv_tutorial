#_*_coding:utf8_*_
import cv2
import numpy as np

#利用霍夫变换在图像中寻找直线，从而实现车道线检测
if __name__ == "__main__":
    image = cv2.imread("lane.jpg")
    image = image[550:, :, :] #取感兴趣区域, 将天空部分去掉
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #转为灰度图
    binary = (gray > 150).astype(np.uint8) * 255   #通过阈值150转为二值图像
    cv2.imshow("binary", binary)
    
    edges = cv2.Canny(binary, 350, 200)            #Canny算子边缘检测
    cv2.imshow("edge", edges)
    lines = cv2.HoughLines(edges, 1, np.pi/90, 200) #霍夫变换
    lines = lines[:, 0, :]
  
    for r, theta in lines:
        k = -1 / np.tan(theta)  #Hesse法线式r = xsin(θ) + ycos(θ)变为直角坐标y = kx + b
        b = r / np.sin(theta)
        #根据y = kx + b，取两个点(0, b), (w, k * w + b) 用来画线
        x1 = 0                  #x = 0是图像最左边的点
        y1 = int(k * x1 + b)
        x2 = image.shape[1] - 1    #image.shape = h, w, c所以x2是图像宽度减一,也就是最右边的点
        y2 = int(k * x2 + b)

        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  #通过两个点将直线画出来
    cv2.imshow("window", image)  #显示图像
    cv2.waitKey(0)
