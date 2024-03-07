import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 基于Haar Cascades的Face Detection
# https://blog.csdn.net/mhhyoucom/article/details/107918472
# 加载定义好的人脸以及眼睛信息匹配信息
face = cv.CascadeClassifier(r'G:\AI-study\class\datas\MNIST\haarcascade_frontalface_default.xml')
eye = cv.CascadeClassifier(r'G:\AI-study\class\datas\MNIST\haarcascade_eye.xml')
# 加载图像
img = cv.imread(r"G:\AI-study\class\datas\MNIST\faces2.png")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 检测图像
faces = face.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    # 画人脸区域
    cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
    # 获得人脸区域
    roi_gray = gray[y:y+h, x:x+w]
    roi_img = img[y:y+h, x:x+w]
    # 检测眼睛
    eyes = eye.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv.rectangle(roi_img, (ex,ey), (ex+w,ey+h),(0,0,255),2)

# 从摄像机获取视频 + 人脸区域提取
# 创建一个基于摄像头的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture(0)
# 遍历获取视频中的图像
# 读取当前时刻的摄像头捕获的图像, 返回为值：True/False, Image/None
success, frame = capture.read()
# 遍历以及等待任意键盘输入
while success and cv.waitKey(1) == -1:
    img = frame
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 直方图均衡化
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # 做一个人脸检测
    # 检测图像
    faces = face.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+h]
        roi_img = img[y:y+h, x:x+h]
        eyes = eye.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv.rectangle(roi_img, (ex,ey), (ex+ew, ey+eh),(0,255,0),2)
    cv.imshow('frame', img)
    # 读取下一帧的图像
    success, frame = capture.read()

cv.imshow('img', img)
cv.waitKey(0)
capture.release()
cv.destroyAllWindows()
