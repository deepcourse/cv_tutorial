#_*_coding:utf8_*_
import numpy as np
import cv2

image = np.zeros((300, 300), dtype = np.uint8)

points = []
for x in range(0, 300, 30): #x = 0, 30, 60 ... 270
    y = int(-0.5 * x + 200)
    if y < 0 or y >= 300:
        continue
    cv2.circle(image, (x, y), 2, (255), 2)
    points.append([x, y])

hough = np.zeros((300, 300), dtype = np.uint8)
for x, y in points:
    #y = k *x + b
    #b = -x * k + y
    #取两个点 (k = -10) (k = 10) 用于作图
    k1 = -10
    b1 = -x * k1 + y
    k2 = 10
    b2 = -x * k2 + y

    k1 *= 40 #缩放x轴40倍，否则会过于集中无法看清
    k2 *= 40
    k1 += 200 #向右移动原点，否则会无法看到交汇点
    k2 += 200
    cv2.line(hough, (k1, b1), (k2, b2), (255), 1)

cv2.imshow("points",image)
cv2.imshow("hough",hough)
cv2.waitKey(0)
