import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img= cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
img1 = img[0:300, 0:200, :]
img2 = img[300:600, 200:400, :]
print(img1.shape, img2.shape)
img3 = 0.3*img1 + 0.7*img2
# img[0:300, 0:200] = img1
# img3 = img1 - img2
# img3 = (img3 + 255)/512.0
# img3 = img3.clip(0,255)
img3 = img3.astype('uint8')
# img3 = img3/255

# 图像的分割与合并
b, g, r = cv.split(img)
img = cv.merge((r, g, b))  # 将原来的r当成新图像的中b，将原来的b当成新图像中的r

# 添加边框
_img = cv.imread(r"G:\AI-study\class\datas\MNIST\opencv-logo.png")

# 1.直接复制
replicate = cv.copyMakeBorder(_img, top=10, bottom=10, left=10, right=10, borderType=cv.BORDER_REPLICATE)
# 2.边界反射
reflect = cv.copyMakeBorder(_img, 10,10,10,10,cv.BORDER_REFLECT)
# 3.边界反射，边界像素不保留
reflect101 = cv.copyMakeBorder(_img, 10,10,10,10,cv.BORDER_REFLECT_101)
# 4.边界延伸循环
wrap = cv.copyMakeBorder(_img, 10,10,10,10, cv.BORDER_WRAP)
# 5.添加常熟
constant = cv.copyMakeBorder(_img,10,10,10,10,cv.BORDER_CONSTANT, value=[128,128,128])

# 可视化
plt.subplot(231)
plt.imshow(cv.cvtColor(_img, cv.COLOR_BGR2RGB))
plt.title("Original")
plt.subplot(232)
plt.imshow(cv.cvtColor(replicate, cv.COLOR_BGR2RGB))
plt.title("Replicate")
plt.subplot(233)
plt.imshow(cv.cvtColor(reflect, cv.COLOR_BGR2RGB))
plt.title("Reflect")
plt.subplot(234)
plt.imshow(cv.cvtColor(reflect101, cv.COLOR_BGR2RGB))
plt.title("Reflect101")
plt.subplot(235)
plt.imshow(cv.cvtColor(wrap, cv.COLOR_BGR2RGB))
plt.title("Wrap")
plt.subplot(236)
plt.imshow(cv.cvtColor(constant, cv.COLOR_BGR2RGB))
plt.title("Constant")
plt.show()


# 图像合并
image1 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
image2 = cv.imread(r"G:\AI-study\class\datas\MNIST\opencv-logo.png")
image1 = cv.resize(image1, (300,300))
image2 = cv.resize(image2, (300,300))
# 添加背景， 计算公式为：dst = alpha * src1 + beta * src2 + gamma
dst = cv.addWeighted(src1=image1, alpha=0.3, src2=image2, beta=1.0, gamma=0)
a = image1*0.3 + image2*1.0
a = np.clip(a, a_min=0, a_max=255).astype(np.uint8)  # 同 cv.addWeighted 方法一致

# 将图片转换为灰度图
img2gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

# 二值化操作： 将输入图像(灰度图像)中所有像素值大于第二个参数的全部设置为第三个参数值
ret, mask = cv.threshold(img2gray, 10, 255, cv.THRESH_BINARY)
# 对图像做一个求反的操作，即255-mask
mask_inv = cv.bitwise_not(mask)

# 获取一个新数据（右上角数据）
_image1 = cv.imread(r"G:\AI-study\class\datas\MNIST\xiaoren.png")
_image2 = cv.imread(r"G:\AI-study\class\datas\MNIST\opencv-logo.png")
rows1, cols1, _ = _image1.shape
rows, cols, channels = _image2.shape
start_rows = 50
end_rows = rows + 50
start_cols = cols1 - cols - 200
end_cols = cols1 - 200
roi = _image1[start_rows:end_rows, start_cols:end_cols]
img2gr = cv.cvtColor(_image2, cv.COLOR_BGR2GRAY)
ret1, mask1 = cv.threshold(img2gr, 10, 255, cv.THRESH_BINARY)
mask1_inv = cv.bitwise_not(mask1)

# 获取得到背景图（对应mask_inv为True的时候，进行and操作，其它位置直接设置为0）
# 在求解bitwise_and操作的时候，如果给定mask的时候，只对mask中对应为1的位置进行and操作，其它位置直接设置为0
img1_bg = cv.bitwise_and(roi, roi, mask=mask1_inv)
# 前景颜色和背景颜色合并
img2_fg = cv.bitwise_and(_image2, _image2, mask=mask1)
dst1 = cv.add(img1_bg, img2_fg)  # dst = img1_bg + img2_fg
_image1[start_rows:end_rows, start_cols:end_cols] = dst1
cv.imshow('res', _image1)
cv.imshow('mask', mask1)
cv.imshow('img1_bg', img1_bg)
cv.imshow('img2_fg', img2_fg)
cv.imshow('dst',dst)
cv.waitKey(0)
cv.destroyAllWindows()




