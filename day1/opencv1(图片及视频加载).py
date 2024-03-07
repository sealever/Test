import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# imread:默认返回结果是: [H,W,C]；当加载的图像是三原色图像的时候，默认返回的通道顺序是: BGR
# NOTE: 给定加载的图像路径不允许有中文，最好也不要有空格，特别是图像文件名称
# 加载图像(如果图像加载失败，那么返回的对象img为None)
# 第一个参数：filename，给定图片路径参数
# 第二个参数：flags，指定图像的读取方式；默认是使用BGR模型加载图像，参考：
# https://docs.opencv.org/3.4.0/d4/da8/group__imgcodecs.html#gga61d9b0126a3e57d9277ac48327799c80af660544735200cbe942eea09232eb822
# 当设置为0表示灰度图像加载，1表示加载BGR图像, 默认为1，-1表示加载alpha透明通道的图像。
img = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
# img = cv.imread(r"G:\AI-study\class\datas\MNIST\koala.png")
print(type(img), img.shape)
print(img[:, :, 0])  # B 通道
print('-' * 100)
print(img[:, :, 1])  # G 通道
print('-' * 100)
print(img[:, :, 2])  # R 通道

# Gray=img[:,:,2]*0.3+img[:,:,1]*0.59+img[:,:,0] *0.11
Gray = img[:, :, 1] * 0.89  # 从一个绿色通道转换成灰度通道 数据类型从uint8转换为float类型，但是取值范围还是[0,255]
Gray = Gray / 255  # 数据范围从[0,255]->[0,1]

#

# # 读取图像将图像转换为Matplotlib可视化# 图像可视化
# # cv.imshow('image', img)
# # # 让图像暂停delay毫秒，当delay秒设置为0的时候，表示永远; 当键盘任意输入的时候，结束暂停
# # cv.waitKey(0)
# # cv.destroyAllWindows()  # 释放所有资源
# # # 明确给定窗口资源
# # cv.namedWindow('image', cv.WINDOW_NORMAL)
# # cv.imshow('image', img)
# # print(cv.waitKey(0))
# # cv.destroyWindow('image') # 释放指定窗口资源
# #
# # # 图像保存 # 第一个参数是图像名称，第二个参数就是图像对象
# # cv.imwrite('koala1', img)
# #
# # # 等待键盘的输入（键盘上各个键对应ASCII码， http://ascii.911cha.com/）
# # k = cv.waitKey(0) & 0xFF
# # if k == 27:
# #     print(k)
# #     cv.destroyAllWindows()
# # else:
# #     # 当输入的是其他键的时候
# #     cv.imwrite('t2.png', img)
# #     cv.destroyAllWindows()
# # NOTE: 如果需要可视化图像，需要注意：OpenCV中是BGR图像，而Matplotlib中是RGB的图像。
# 将BRG 转换为RGB（方式一）
# img2 = np.zeros_like(img, dtype=img.dtype)
# img2[:, :, 0] = img[:, :, 2]
# img2[:, :, 1] = img[:, :, 1]
# img2[:, :, 2] = img[:, :, 0]
# 将BRG 转换为RGB（方式二）
# img2 = img[:, :, ::-1]
# # plt.imshow(img, cmap='gray')
# plt.imshow(img2, cmap='gray')
# plt.show()

#  视频基本处理
# 从摄像机获取视频
# 创建一个基于摄像头的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture(0)
# 设置摄像头相关参数
# success = capture.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
# if success:
#     print('设置宽度成功')
# success = capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
# if success:
#     print('设置高度成功')

# 打印属性
size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)
# 此时摄像头的帧率(摄像头图像数据没有产生，没办法指导帧率)
print(capture.get(cv.CAP_PROP_FPS))
# 创建一个视频输出对象
# 设置视频中的帧率，也就是每秒存在多少张图片
fps = 30
# VideoWriter中的size为（宽度，高度），需保证一致，否则保存视频无法打开
video_writer = cv.VideoWriter('cb.avi', cv.VideoWriter_fourcc(*'DIVX'), fps, size)
# 构建10s的视频输出
num = 10 * fps - 1
# 遍历获取视频中的图像
# 读取当前时刻的摄像头捕获的图像, 返回为值：True/False, Image/None
success, frame = capture.read()
# print(type(frame), frame.shape)
# 遍历以及等待任意键盘输入（-1表示等待delay后，没有任何键盘输入 --> 没有键盘输入的情况下，就继续循环处理）
# while success and cv.waitKey(delay=1) == -1:
#     cv.imshow('frame', frame)
#     # 读取下一帧的图像
#     success, frame = capture.read()
# 10s 录像
while success and num > 0:
    video_writer.write(frame)
    success, frame = capture.read()
    num -= 1

import os
print(os.getcwd())
# 释放资源
capture.release()
cv.destroyAllWindows()

# 视频读取
p = r'G:\AI-study\class\datas\MNIST\img'
if not os.path.exists(p):
    os.makedirs(p)
# 视频文件读入
# 创建一个基于文件的视频读取流，给定基于第一个视频设备
capture = cv.VideoCapture("cb.avi")
size = (int(capture.get(cv.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv.CAP_PROP_FRAME_HEIGHT)))
print(size)
print(capture.get(cv.CAP_PROP_FPS))
# 读取当前时刻的摄像头捕获的图像, 返回为值：True/False, Image/None
success, frame = capture.read()
# 遍历以及等待任意键盘输入
k = 0
while success and cv.waitKey(60) == -1:
    if k % 100 == 0:
        cv.imwrite(r'G:\AI-study\class\datas\MNIST\img\img_{}.png'.format(k), frame)
    cv.imshow('frame', frame)
    k += 1
    success, frame = capture.read()
capture.release()
cv.destroyAllWindows()