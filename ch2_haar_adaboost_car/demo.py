import cv2
import os

cascade = cv2.CascadeClassifier("save_dir/cascade.xml")

files = os.listdir("dataset/test_img")
for f in files:
    print f
    img = cv2.imread("dataset/test_img/" + f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in cars:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    cv2.imshow('window', img)
    cv2.waitKey(0)
