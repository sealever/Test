import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 直方图
# OpenCV中的直方图的主要功能是可以查看图像的像素信息
# 以及提取直方图中各个区间的像素值的数目作为当前图像的特征属性进行机器学习模型
img = cv.imread(r"G:\AI-study\class\datas\MNIST\tietu.png")
# 基于OpenCV的API计算直方图
his1 = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256])
# 基于Numpy计算直方图
his2, bins = np.histogram(img.ravel(), 256, [0,256])
# 和np.histogram一样的计算方式，但是效率快10倍
his3 = np.bincount(img.ravel(), minlength=256)
plt.figure(figsize=(20,10))
plt.subplot(221)
plt.imshow(img, 'gray')
plt.title('原图')

plt.subplot(222)
plt.plot(his1)
plt.title('his1')

plt.subplot(223)
plt.plot(his2)
plt.title('his2')

plt.subplot(224)
plt.plot(his3)
plt.title('his3')

plt.figure(figsize=(20,10))
# 针对彩色图像计算直方图
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv.calcHist([img], [i], None, [256], [0,256])
    plt.plot(histr, color=col)
    plt.xlim([0,256])

# 创建一个mask
mask = np.zeros(img.shape[:2], np.uint8)
mask[50:450, 50:450] = 255
# 构建mask区域的图像
mask_img = cv.bitwise_and(img, img, mask=mask)
# 计算直方图
hist1 = cv.calcHist([img], [0], None, [256], [0,256])
hist2 = cv.calcHist([mask_img], [0], mask, [256], [0,256])
plt.figure(figsize=(20,10))
# 可视化
plt.subplot(221)
plt.imshow(img, 'gray')
plt.title('Original Image')

plt.subplot(222)
plt.imshow(mask_img, 'gray')
plt.title('masked_img')

plt.subplot(223)
plt.plot(hist1, color='r')
plt.plot(hist2, color='g')
plt.title('Hist')

# 直方图均衡化
# 如果某一个图像的像素仅限制于某个特定的范围，但是实际上一个比较好的图像像素点
# 应该是在某一个范围内，所以需要将像素的直方图做一个拉伸的操作；比如：一个偏暗
# 的图像，像素基本上都在较小的位置，如果将像素值增大，不就可以让图像变亮嘛！
# 实际上，直方图均衡化操作是一个提高图像对比度的方式
img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img1 = cv.GaussianBlur(img1, (5,5), 0)
img2 = cv.equalizeHist(img1)
# 做一个自适应的直方图均衡
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img3 = clahe.apply(img1)
plt.figure(figsize=(20,10))
# 可视化
plt.subplot(221)
plt.imshow(img1, 'gray')
plt.title('Original Image')

plt.subplot(222)
plt.imshow(img2, 'gray')
plt.title('img2')

plt.subplot(223)
plt.imshow(img3, 'gray')
plt.title('img3')

plt.subplot(224)
plt.plot(cv.calcHist([img1], [0], None, [256], [0,256]),color='r', label='img1' )
plt.plot(cv.calcHist([img2], [0], None, [256], [0,256]), color='g',label='img2')
plt.plot(cv.calcHist([img3], [0], None, [256], [0,256]), color='b', label='img3')
plt.legend(loc='upper left')
plt.title('Hist')

# 模板匹配
plt.figure(figsize=(16,16))
img4 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png",0)
img5 = img4.copy()
# 加载模板图像
template = cv.imread(r"G:\AI-study\class\datas\MNIST\template.png",0)
# 得到模板图像的高度和宽度
h, w = template.shape
# 6种匹配方式
methods = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 'cv.TM_CCORR_NORMED', 'cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']
img5 = img5[:,:400]
for idx, meth in enumerate(methods):
    img6 = img5.copy()
    method = eval(meth)  # 得到对应的方式(eval的意思是执行)
    # 使用给定的方式进行模板匹配（返回值为各个局部区域和模板template之间的相似度<可以认为是相似度>）
    res = cv.matchTemplate(img6,template, method)
    # 从数据中查找全局最小值和最大值, 以及对应的位置
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    # 如果求解方式为TM_SQDIFF和TM_SQDIFF_NORMED，
    # 那么矩形左上角的点就是最小值的位置；
    # 否则是最大值的位置
    if method in [cv.TM_CCORR, cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    # 基于左上角的坐标计算右下角的坐标
    bottom_right = (top_left[0] + w, top_left[1] + h)
    # 基于坐标画矩形
    cv.rectangle(img6, top_left, bottom_right, 180, 5)
    # 画图
    plt.subplot(len(methods) // 2, 4, 2 * idx + 1)
    plt.imshow(res, cmap='gray')
    plt.title('Matching Result {}'.format(meth))
    plt.subplot(len(methods) // 2, 4, 2 * idx + 2)
    plt.imshow(img6, cmap='gray')
    plt.title('Detected Point {}'.format(meth))
    plt.suptitle(meth)



plt.show()

