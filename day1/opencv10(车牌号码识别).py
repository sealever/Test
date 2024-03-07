import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# 做一个简单的车牌号码提取的代码
# - 方式一：基于关键点检测 + 透视变换
# - 方式二：基于二值化处理 + 轮廓处理 + 边缘处理

# 图像预处理
def preprocess(img):
    # 高斯平滑
    gaussian = cv.GaussianBlur(img, (3, 3), 0, 0, cv.BORDER_DEFAULT)
    # 中值滤波
    median = cv.medianBlur(gaussian, 5)
    # sobel 提取边缘体征（卷积）
    sobel = cv.Sobel(median, cv.CV_64F, dx=1, dy=0, ksize=3)
    sobel = np.uint8(np.absolute(sobel))

    # 二值化
    ret, binary = cv.threshold(sobel, 170, 255, cv.THRESH_BINARY)
    # 膨胀&腐蚀
    element1 = cv.getStructuringElement(cv.MORPH_RECT, (9, 1))  # 形态学的核kernel,矩形
    element2 = cv.getStructuringElement(cv.MORPH_RECT, (9, 7))  # 形态学的核kernel,矩形
    # 膨胀
    dilation = cv.dilate(binary, element2, iterations=1)
    # 腐蚀
    erosion = cv.erode(dilation, element1, iterations=2)
    # 膨胀
    dilation2 = cv.dilate(erosion, element2, iterations=5)
    # 腐蚀
    erosion2 = cv.erode(dilation2, element1, iterations=4)
    return erosion2


# 进行车牌照区域查找
def find(img):
    # 查找轮廓(img: 原始图像，contours：矩形坐标点，hierarchy：图像层次)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # 查找矩形
    max_ratio = -1
    max_box = None
    ratios = []
    number = 0
    for i in range(len(contours)):
        cnt = contours[i]  # 当前轮廓的坐标信息
        # 计算轮廓面积
        area = cv.contourArea(cnt)
        # 面积太小的过滤掉
        if area < 10000:
            continue
        # 找到最小的矩形
        rect = cv.minAreaRect(cnt)
        # 矩形的四个坐标（顺序不定，但是一定是一个左下角、左上角、右上角、右下角这种循环顺序(开始是哪个点未知)）
        box = cv.boxPoints(rect)
        # 转换为long 类型
        box = np.int64(box)
        # 计算长宽高
        # 第一条边的高
        a = abs(box[0][0] - box[1][0])
        b = abs(box[0][1] - box[1][1])
        d1 = np.sqrt(a ** 2 + b ** 2)
        # 第二条边的长度
        c = abs(box[1][0] - box[2][0])
        d = abs(box[1][1] - box[2][1])
        d2 = np.sqrt(c ** 2 + d ** 2)
        # 让最小值为高度，最大值为宽度
        height = int(min(d1, d2))
        weight = int(max(d1, d2))
        # 计算面积
        area2 = height * weight
        # 两个面积的差值一定在一定范围内
        r = np.absolute((area2 - area) / area)
        if r > 0.6:
            continue
        ratio = float(weight) / float(height)
        cv.drawContours(img, [box], 0, 255, 2)
        cv.imshow('img', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
        # 实际情况下ratio应该是3左右，但是由于我们的照片不规范的问题，
        # 检测出来的宽度比高度应该在2~5.5之间
        if ratio > max_ratio:
            max_box = box
            max_ratio = ratio
        if ratio > 5.5 or ratio < 2:
            continue
        number += 1
        ratios.append((box, ratio))
    # 根据找到的图像矩阵数量进行数据输出
    print("总共找到:{}个可能区域!!".format(number))
    if number == 1:
        return ratios[0][0]
    elif number > 1:
        # 不考虑太多，直接获取中间值(并且进行过滤)
        # 实际要求更加严格
        filter_ratios = list(filter(lambda t: 2.7 <= t[1] < 5, ratios))
        size = len(filter_ratios)
        if size == 1:
            return filter_ratios[0][0]
        elif size > 1:
            # 取中间值
            ratios1 = [filter_ratios[i][1] for i in range(size)]
            ratios1 = list(zip(range(size), ratios1))
            # 数据排序
            ratios1 = sorted(ratios1, key=lambda t: t[1])
            # 获取中间值的数据
            idx = ratios[size // 2][0]
            return filter_ratios[idx][0]
        else:
            # 获取最大值
            ratios1 = [ratios[i][1] for i in range(number)]
            ratios1 = list(zip(range(number), ratios1))
            # 数据排序
            ratios1 = sorted(ratios1, key=lambda t: t[1])
            # 获取最大值的数据
            idx = ratios1[-1][0]
            return filter_ratios[idx][0]
    else:
        # 直接返回最大值
        print("直接返回最接近比例的区域...")
        return max_box


# 截取车牌区域
def cut(path):
    img = cv.imread(path)
    h, w, _ = img.shape
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 图像预处理-->将车牌区域给明显的显示出来
    img1 = preprocess(img)
    # 查找车牌区域(假定只会有一个)
    box = find(img1)
    # 返回区域对应的图像
    # 因为不知道，点的顺序，所以对左边点坐标排序
    ys = [box[0][1], box[1][1], box[2][1], box[3][1]]
    xs = [box[0][0], box[1][0], box[2][0], box[3][0]]
    ys_index = np.argsort(ys)
    xs_index = np.argsort(xs)
    # 获取x上的坐标
    x1 = box[xs_index[0]][0]
    x1 = x1 if x1 > 0 else 0
    x2 = box[xs_index[3]][0]
    x2 = w if x2 > w else x2
    # 获取y上的坐标
    y1 = box[ys_index[0]][1]
    y1 = y1 if y1 > 0 else 0
    y2 = box[ys_index[3]][1]
    y2 = h if y2 > h else y2
    # 截取图像
    img_plate = img[y1:y2, x1:x2]
    return img_plate


if __name__ == '__main__':
    path = r'G:\AI-study\class\datas\MNIST\car2.jpg'
    cut_img = cut(path)
    cv.imshow('image', cv.imread(path))
    cv.imshow('plat', cut_img)
    cv.waitKey(0)
    cv.destroyAllWindows()