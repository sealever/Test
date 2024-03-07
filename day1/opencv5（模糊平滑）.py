import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
# 自定义一个kernel核 （1/9均值）
# kernel = np.ones((3,3), np.float32) / 9
kernel = np.array([
    [1, 2, 1],
    [0, -8, 0],
    [1, 2, 1]
], dtype=np.float32)
# 做一个卷积操作
# 第二个参数为：ddepth，一般为-1，表示不限制，默认值即可。
img2 = cv.filter2D(img1, -1, kernel)

# 大小缩放
h, w, _ = img1.shape
h = int(h * 0.5)
w = int(w * 0.5)
img3 = cv.resize(img1, (w, h))
img4 = cv.resize(img2, (w, h))

w = int(w * 0.2)
h = int(h * 0.2)
img5 = cv.resize(img3, (w, h))
img6 = cv.resize(cv.filter2D(img4, -1, kernel), (w, h))
images = [img1, img3, img5, img2, img4, img6]
for i in range(6):
    images[i] = cv.cvtColor(images[i], cv.COLOR_BGR2RGB)
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i])
plt.show()

# 自定义卷积
# ksize：给定窗口大小
# sigmaX: 给定横向的kernel中，参数的标准差
# sigmaY: 给定纵向的kernel中，参数的标准差
image = cv.GaussianBlur(img1, ksize=(5, 5), sigmaX=2.0, sigmaY=2.0)
plt.figure(figsize=(20, 10))
plt.subplot(2, 6, 1)
plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
plt.subplot(2, 6, 7)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
for i in range(5):
    h, w, _ = img1.shape
    w = int(w * 0.5)
    h = int(h * 0.5)
    g_img = cv.GaussianBlur(image, ksize=(5, 5), sigmaX=2.0, sigmaY=2.0)
    img1 = cv.resize(img1, (w, h))
    image = cv.resize(g_img, (w, h))
    plt.subplot(2, 6, i + 2)
    plt.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB))
    plt.subplot(2, 6, i + 8)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.show()

# 均值滤波 :选择窗口中的所有值的均值作为输出值
img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\koala.png")
dst = cv.blur(img1, ksize=(11, 11))

# 高斯滤波
# 窗口滤波过程中对应的卷积参数是通过高斯函数计算出来的，特点是中间区域的权重系数大，而周边区域的权重系数小。
# 作用：去除高斯噪声数据以及图像模糊化操作
# 查看5*5的高斯卷积kernel
# 窗口大小5*5，标准差为1 ,cv.CV_64F为64位浮点数
a = cv.getGaussianKernel(9, 2, ktype=cv.CV_64F)
b = np.transpose(a)
dst1 = cv.filter2D(cv.filter2D(img1, -1, a), -1, b)

# 中值滤波： 选择窗口区域中的中值作为输出值
img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
noisy = np.random.normal(10, 10, (img1.shape[0], img1.shape[1], img1.shape[2]))
noisy = np.clip(noisy, 0, 255).astype(np.uint8)
img1 = img1 + noisy
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
dst2 = cv.medianBlur(img1, ksize=5)  # 中值滤波

plt.subplot(121)
plt.imshow(img1, 'gray')
plt.title('原图')
plt.subplot(122)
plt.imshow(dst2, 'gray')
plt.title('中值滤波')
plt.show()

# # 双边滤波: 删除中间的纹理，保留边缘信息
img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
dst3 = cv.bilateralFilter(img1, d=9, sigmaColor=75, sigmaSpace=75)  # 双边滤波
plt.subplot(121)
plt.imshow(img1, 'gray')
plt.title('原图')
plt.subplot(122)
plt.imshow(dst3, 'gray')
plt.title('双边滤波')
plt.show()
