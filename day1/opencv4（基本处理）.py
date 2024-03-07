import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取所有的颜色通道(以COLOR_开头的属性)
# 在OpenCV中HSV颜色空间的取值范围为：H->[0,179], S->[0,255], V->[0,255]; 其它图像处理软件不一样
flag = [i for i in dir(cv) if i.startswith('COLOR_')]
print("总颜色转换方式:{}".format(len(flag)))

img = cv.imread(r"G:\AI-study\class\datas\MNIST\opencv-logo.png")
# 转换为灰度图
# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 转换为HSV 格式
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
print(hsv.transpose(2,0,1))
print("=" * 100)
print(img.transpose(2,0,1))
# 定义像素点范围
# # 蓝色的范围
# lower = np.array([100,50,50])
# upper = np.array([130,255,255])
# 红色的范围
lower1 = np.array([150,50,50])
upper1 = np.array([200,255,255])
# 绿色的范围
lower2 = np.array([30,50,50])
upper2 = np.array([80,255,255])
# 在这个范围的图像像素设置为255，不在这个范围的设置为0
mask1 = cv.inRange(hsv, lower1, upper1)
mask2 = cv.inRange(hsv, lower2, upper2)
mask = cv.add(mask1, mask2)
# 进行And操作进行数据合并
dst = cv.bitwise_and(img, img, mask=mask2)

# 大小重置
img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
old_height, old_weight, _ = img1.shape
new_height = int(0.8 * old_height)
new_weight = int(0.8 * old_weight)
dst1 = cv.resize(img1, (new_weight, new_height))
print(dst1.shape)

# 图像平移
M = np.float32([[1,0,50],   # 水平方向往右平移50个像素  m11*x+m12*y+m13
                [0,1,-10]])  # 垂直方向往上平移10个像素 m21*x+m22*y+m23
# warpAffine计算规则：src(x,y)=dst(m11*x+m12*y+m13, m21*x+m22*y+m23)  x和y表示的是原始图像中的坐标
# x和y是坐标点
dst2 = cv.warpAffine(img1, M, (old_weight, old_height))

# 图像旋转
# 构建一个用于旋转的R(旋转的中心点，旋转大小，尺度)
# angle:负数表示顺时针选择 borderValue背景颜色
R = cv.getRotationMatrix2D(center=(old_height/2, old_weight/2), angle=30, scale=0.5)
dst3 = cv.warpAffine(img1, R, (old_weight, old_height), borderValue=[0,0,0])
R = cv.getRotationMatrix2D(center=(0,0), angle=20, scale=0.5)
dst4 = cv.warpAffine(dst3, R, (old_weight, old_height), borderValue=[100, 154, 38])

# 旋转方式二
dst5 = cv.rotate(img1, rotateCode=cv.ROTATE_90_CLOCKWISE)  # 顺时针90度
dst6 = cv.rotate(img1, rotateCode=cv.ROTATE_90_COUNTERCLOCKWISE)  # 逆时针90度

# 垂直或水平翻转（镜像）
dst7 = cv.flip(img1, 0)  # 上下翻转
dst8 = cv.flip(img1, 1)  # 左右翻转

# 图像旋转变成水平一点
img2 = cv.imread(r"G:\AI-study\class\datas\MNIST\car3_plat.jpg")
h, w, _ = img2.shape
M = cv.getRotationMatrix2D(center=(0,0), angle=20, scale=1)
dst9 = cv.warpAffine(img2, M, (w+30, w//2), borderValue=[100, 154, 38])


# 仿射变换
#  在仿射变换中，原图中是平行的元素在新的图像中也是平行的元素；可以任意的给定三个点来构建
p1 = np.float32([[170,200], [350,200], [170,400]])
p2 = np.float32([[10,50], [200,50], [100,300]])
M = cv.getAffineTransform(p1, p2)
dst10 = cv.warpAffine(img1, M, (old_weight, old_height))

# 透视变换
# 实际上就是根据给定的四个点来进行转换操作，在转换过程中图像的形状不会发现变化
# 也就是原来是直线的，转换后还是直线，要求这四个点中任意三个点均不在同一线上
cv.line(img1, pt1=(0,old_height//2), pt2=(old_weight,old_height//2), color=(255,0,0), thickness=5)
cv.line(img1, pt1=(old_weight//2,0), pt2=(old_weight//2,old_height), color=(255,0,0), thickness=5)
p1 = np.float32([[56,65], [368,52], [28,387], [389,390]])
p2 = np.float32([[0,0], [300,0], [0,300], [300,300]])
R1 = cv.getPerspectiveTransform(p1, p2)  # M是一个3*3的矩阵
# 计算规则：src(x,y)=dst((m11x+m12y+m13)/(m31x+m32y+m33),
#                    (m21x+m22y+m23)/(m31x+m32y+m33))
dst11 = cv.warpPerspective(img1, R1, (300,300))


# 二值化图像
img3 = np.arange(255, -1, -1).reshape(1, -1)
for i in range(255):
    img3 = np.append(img3, np.arange(255, -1, -1).reshape(1, -1), axis=0)
img3 = img3.astype(np.uint8)
img5 = cv.imread(r"G:\AI-study\class\datas\MNIST\tietu.png")
img4 = cv.cvtColor(img5, cv.COLOR_BGR2GRAY)
# 进行普通二值化操作(第一个参数是返回的阈值，第二个参数返回的是二值化之后的图像)
# 普通二值化操作， 将小于等于阈值thresh的设置为0，大于该值的设置为maxval
ret1, thresh1 = cv.threshold(src=img4, thresh=127, maxval=152, type=cv.THRESH_BINARY)

# 反转二值化操作，将小于等于阈值thresh的设置为maxval，大于该值的设置为0
ret2, thresh2 = cv.threshold(img4, thresh=127, maxval=152, type=cv.THRESH_BINARY_INV)

# 截断二值化操作，将小于等于阈值thresh的设置为原始值，大于该值的设置为maxval
ret3, thresh3 = cv.threshold(img4, 127, 210, cv.THRESH_TRUNC)

# 0二值化操作，将小于等于阈值的设置为0，大于该值的设置为原始值
ret4, thresh4 = cv.threshold(img4, 127, 255, cv.THRESH_TOZERO)

# 反转0二值化操作，将小于等于阈值的设置为原始值，大于阈值的设置为0
ret5, thresh5 = cv.threshold(img4, 127, 255, cv.THRESH_TOZERO_INV)

# 自适应均值二值化，使用均值的方式产生当前像素点对应的阈值，
# 使用(x,y)像素点邻近的blockSize*blockSize区域的均值寄减去C的值
th2 = cv.adaptiveThreshold(img4, maxValue=255, adaptiveMethod=cv.ADAPTIVE_THRESH_MEAN_C,
                           thresholdType=cv.THRESH_BINARY, blockSize=11, C=2)

# 自适应高斯二值化，高斯（模糊化）使用高斯分布的方式产生当前像素点对应的阈值
# 使用(x,y)像素点邻近的blockSize*blockSize区域的加权均值寄减去C的值，
# 其中权重为和当前数据有关的高斯随机数
th3 = cv.adaptiveThreshold(img4, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)

# 大津法二值化
re2, th4 = cv.threshold(img4, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 高斯 + 大津法二值化
blur = cv.GaussianBlur(img4, (5,5), 0)
re3, th5 = cv.threshold(blur, 0,255, cv.THRESH_BINARY+cv.THRESH_OTSU)

# 自适应高斯 + 高斯 + 大津法二值化
ret, th6 = cv.threshold(cv.GaussianBlur(th3,(5,5), 0), 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)

titles = ['原图','二值化','反二值化','截断法','0二值化','反0二值', '自适应均值', '自适应高斯', '大津法', '高斯+大津', '自高+高+大']
images = [img4, thresh1, thresh2, thresh3, thresh4, thresh5, th2, th3, th4, th5, th6]
for i in range(11):
    plt.subplot(3,4,i+1)
    plt.imshow(images[i]/255.0,'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# 产生噪声数据
img6 = np.random.uniform(low=0, high=255, size=(300,300))
# img6 = np.random.normal(150, 100, size=(300,300))
img6 = np.clip(img6, 0, 255)
img6 = img6.astype(np.uint8)
# 产生背景图像
img7 = np.zeros((300,300), dtype=np.uint8)
img7[100:200, 100:200] = 255
# 合并图像
img8 = cv.addWeighted(img6, alpha=0.3, src2=img7, beta=0.3, gamma=0)

cv.imshow('image', img8)
cv.waitKey(0)
cv.destroyAllWindows()