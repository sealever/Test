import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import copy

#  形态学转换
# 主要包括腐蚀、扩张、打开、关闭等操作；主要操作是基于kernel核的操作
# 常见的核主要有：矩阵、十字架、椭圆结构的kernel
kernel1 = cv.getStructuringElement(cv.MORPH_RECT,ksize=(5,5))
print("矩形：\n{}".format(kernel1))
kernel2 = cv.getStructuringElement(cv.MARKER_CROSS, ksize=(5,5))
print("十字架：\n{}".format(kernel2))
kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE,ksize=(5,5))
print("椭圆：\n{}".format(kernel3))
kernel = np.ones((9,9), np.uint8)
# 腐蚀
# 腐蚀的意思是将边缘的像素点进行一些去除的操作；腐蚀的操作过程就是让kernel核在图像上进行滑动，当内核中的所有像素被视为1时，原始图像中对应位置的像素设置为1，否则设置为0.
# 其主要效果是：可以在图像中减少前景图像(白色区域)的厚度，有助于减少白色噪音，可以用于分离两个连接的对象
# 一般应用与只有黑白像素的灰度情况
img = cv.imread(r"G:\AI-study\class\datas\MNIST\j.png")
# a. 定义一个核(全部设置为1表示对核中5*5区域的所有像素均进行考虑，设置为0表示不考虑)
# 核的定义和卷积不一样，卷积里面是参数的意思，膨胀里面是范围的意思(1表示包含，0表示不考虑)
kernel1 = np.ones((5,5), np.uint8)
# 方式一
dst1 = cv.erode(img, kernel1, iterations=1, borderValue=cv.BORDER_REFLECT)
# 方式二
dst2 = cv.morphologyEx(img, cv.MORPH_ERODE, kernel3)

# 扩张/膨胀
# 和腐蚀的操作相反，其功能是增加图像的白色区域的值,只要在kernel中所有像素中有可以视为1的像素值，
# 那么就将原始图像中对应位置的像素值设置为1，否则设置为0。通常情况下，在去除噪音后，可以通过扩张在恢复图像的目标区域信息。
# # 方式一
dst3 = cv.dilate(img, kernel1, iterations=1)
# 方式二
dst4 = cv.morphologyEx(img, cv.MORPH_DILATE, kernel2)


# 加载白色噪声数据
h, w, _ = img.shape
for i in range(100):
    x = np.random.randint(h)
    y = np.random.randint(w)
    img[x, y] = 255
# 加载黑色噪声数据
for i in range(100):
    x = np.random.randint(h)
    y = np.random.randint(w)
    img[x, y] = 0
# # Open:一次腐蚀 + 一次膨胀，一般用于去除白色噪音数据。
dst5 = cv.morphologyEx(img, cv.MORPH_OPEN, kernel1, iterations=1)
# closing : 一次膨胀 + 一次腐蚀，一般用于去除黑色噪音数据。
dst6 = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel1, iterations=1)
img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\j.png")
# Morphological Gradient 形态梯度: 膨胀 - 腐蚀
dst7 = cv.morphologyEx(img1, cv.MORPH_GRADIENT, kernel1, iterations=1)
# Top Hat: 原图 - open (减)
dst8 = cv.morphologyEx(img1, cv.MORPH_TOPHAT, kernel, iterations=1)
# Block Hat: 原图 - close(减)
dst9 = cv.morphologyEx(img1, cv.MORPH_BLACKHAT, kernel, iterations=1)

# 图像梯度
# 通过对图像梯度的操作，提取图像的边缘信息
# 在OpenCV中提供了三种类型的高通滤波器，常见处理方式：Sobel、Scharr以及Laplacian导数
# sobel：Sobel滤波器通过加入高斯平滑以及方向导数，抗噪性比较强。
img2 = cv.imread(r"G:\AI-study\class\datas\MNIST\tietu.png")
img2 = cv.resize(img2,(500, 500))
img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
# 画几条线条
# rows, cols, _ = img2.shape
# cv.line(img2, pt1=(0,rows//3), pt2=(cols,rows//3), color=0, thickness=5)
# cv.line(img2, pt1=(0,2*rows//3), pt2=(cols,2*rows//3), color=0, thickness=5)
# cv.line(img2, pt1=(cols//3,0), pt2=(cols//3,rows), color=0, thickness=5)
# cv.line(img2, pt1=(2*cols//3,0), pt2=(2*cols//3,rows), color=0, thickness=5)
# cv.line(img2, pt1=(0,0), pt2=(cols,rows), color=0, thickness=5)
# cv.line(img2, pt1=(0,rows), pt2=(cols,0), color=0, thickness=5)
# print("")
# x方向的的Sobel过滤，ksize：一般取值为3,5,7；
# 第二个参数：ddepth，给定输出的数据类型的取值范围，默认为unit8的，取值为[0,255]，如果给定-1，表示输出的数据类型和输入一致。
sobel_x = cv.Sobel(img2, -1, dx=1, dy=0, ksize=5)
sobel_y = cv.Sobel(img2, cv.CV_64F, dx=0, dy=1, ksize=5)
sobel_x2 = cv.Sobel(img2, cv.CV_64F, dx=2, dy=0, ksize=5)
sobel_y2 = cv.Sobel(img2, cv.CV_64F, dx=0, dy=2, ksize=5)
sobel = cv.Sobel(img2, cv.CV_64F, dx=1, dy=1, ksize=5)
sobel_xy = cv.Sobel(sobel_x, cv.CV_64F, dx=0, dy=1, ksize=5)
sobel_yx = cv.Sobel(sobel_y, cv.CV_64F, dx=1, dy=0, ksize=5)

# Scharr: Scharr可以认为是一种特殊的Sobel方式, 实际上就是一种特殊的kernel
# Scharr中，dx和dy必须有一个为0，一个为1
scharr_x = cv.Scharr(img2, -1, dx=1, dy=0)
scharr_y = cv.Scharr(img2, -1, dx=0, dy=1)
scharr_xy = cv.Scharr(scharr_x, -1, dx=0, dy=1)
scharr_yx = cv.Scharr(scharr_y, -1, dx=1, dy=0)

# Laplacian 使用拉普拉斯算子进行边缘提取
laplacian = cv.Laplacian(img2, -1, ksize=5)
# 绝对值化：将梯度负值转为正值。在Sobel检测中，当输出的depth设置
# 为比较低的数据格式，那么当梯度值计算为负值的时候，就会将其重置为0，从而导致失真。
# 在Laplacian检测中，该问题不大
abs_l = np.uint8(np.absolute(laplacian))

# Canny: 算法是一种比Sobel和Laplacian效果更好的一种边缘检测算法
# 使用高斯滤波：降噪，使用5*5的kernel做Gaussian filter降噪；
# 计算梯度：求图像每个像素的梯度大小和方向；
# 非极大值抑制：删除可能不构成边缘的像素，即在渐变方向上相邻区域的像素梯度值是否是最大值，如果不是，则进行删除。
# 双阈值判别：基于阈值来判断是否属于边；大于max的一定属于边缘，小于min的一定不属于边缘，在这个中间的可能属于边的边缘。
# 抑制孤立的弱边缘
canny = cv.Canny(img2, threshold1=50, threshold2=250)
images = [img2, sobel_x, sobel_y, sobel_x2, sobel_y2, sobel, sobel_xy, sobel_yx, scharr_x, scharr_y, scharr_xy, scharr_yx, laplacian,abs_l, canny]
names = ['img2', 'sobel_x', 'sobel_y', 'sobel_x2', 'sobel_y2', 'sobel', 'sobel_xy', 'sobel_yx','scharr_x', 'scharr_y', 'scharr_xy', 'scharr_yx','laplacian', 'abs_l', 'canny']
plt.figure(1,figsize=(20,20))
for i in range(15):
    plt.subplot(4,4,i+1)
    plt.imshow(images[i])
    plt.title(names[i])



# 轮廓信息
# 轮廓信息可以简单的理解为图像曲线的连接点信息，在目标检测以及识别中有一定的作用。
# 轮廓信息的查找最好是基于灰度图像或者边缘特征图像，因为基于这样的图像比较容易找连接点信息；
# NOTE:在OpenCV中，查找轮廓是在黑色背景中查找白色图像的轮廓信息。
img3 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png",0)
img4 = cv.bitwise_not(img3)  # 图像反转
ret, thresh = cv.threshold(img3, 127,255,cv.THRESH_BINARY)
# 发现轮廓信息
# 第一个参数是原始图像，第二个参数是轮廓的检索模型，第三个参数是轮廓的近似方法
# 第一个返回值为轮廓，第三个返回值为层次信息
# CHAIN_APPROX_SIMPLE指的是对于一条直线上的点而言，仅仅保留端点信息,
# 而CHAIN_APPROX_NONE保留所有点
# contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
# contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
# 在图像中绘制边缘
# 当contourIdx为-1的时候，表示绘制所有轮廓，当为大于等于0的时候，表示仅仅绘制某一个轮廓
# 这里的返回值img3和img是同一个对象，在当前版本中
img5 = cv.drawContours(copy.deepcopy(img3), contours, contourIdx=-1, color=(0,0,255), thickness=5)
max_idx = np.argmax([len(t) for t in contours])
img5 = cv.drawContours(img3, contours,contourIdx=-1, color=(0,255,0), thickness=2)
# 计算轮廓面积，周长
cnt = contours[max_idx]
area = cv.contourArea(cnt)  # 面积
perimeter = cv.arcLength(cnt, closed=True)
print("面积为:{}, 周长为:{}".format(area, perimeter))
# 获取最大的矩形边缘框
x, y, w, h = cv.boundingRect(cnt)
cv.rectangle(img3, pt1=(x,y), pt2=(x+w,y+h),color=(255,0,0), thickness=2)
# 绘制最小矩形(所有边缘在矩形内)、得到矩形的点(左下角、左上角、右上角、右下角<顺序不一定>)、绘图
# minAreaRect：求得一个包含点集cnt的最小面积的矩形，这个矩形可以有一点的旋转偏转的，输出为矩形的四个坐标点
# rect为三元组，第一个元素为旋转中心点的坐标
# rect为三元组，第二个元素为矩形的高度和宽度
# rect为三元组，第三个元素为旋转大小，正数表示顺时针选择，负数表示逆时针旋转
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int64(box)
cv.drawContours(img3, [box], 0, (0,255,0), 2)
# 绘制最小的圆（所有边缘在圆内）
(x, y), radius = cv.minEnclosingCircle(cnt)
center = (int(x), int(y))
radius = int(radius)
cv.circle(img5, center, radius, (0,0,255),5)
# 绘制最小的椭圆（所有边缘不一定均在圆内）
ellipsis = cv.fitEllipse(cnt)
cv.ellipse(img3, ellipsis, (0,255,0), 0)


# 将轮廓当成边缘
img6 = np.zeros_like(img5)
cv.drawContours(img6, contours, contourIdx=max_idx, color=(255,0,0),thickness=2)
plt.figure(2,figsize=(20,20))
images1 = [img3, img4, thresh, img5, img6]
titles = ['原图', '反转', '二值化', '轮廓', '边缘']
for i in range(5):
    plt.subplot(2,3,i+1)
    plt.imshow(images1[i], 'gray')
    plt.title(titles[i])
plt.show()



cv.imshow('dst', img2)
cv.waitKey(0)
cv.destroyAllWindows()