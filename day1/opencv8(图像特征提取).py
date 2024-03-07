import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 方式一：直接将图像原始特征扁平化
img1 = cv.imread(r"G:\AI-study\class\datas\MNIST\tietu.png", 0)
feature = img1.reshape(1, -1)
print(feature.shape)

# 方式二：提取直方图特征,有可能需要提取不同颜色空间的直方图特征，然后再合并
img = cv.imread(r"G:\AI-study\class\datas\MNIST\tietu.png")
feature1, _ = np.histogram(cv.cvtColor(img, cv.COLOR_BGR2RGB).ravel(), bins=16)
feature2, _ = np.histogram(cv.cvtColor(img, cv.COLOR_BGR2GRAY).ravel(), bins=16)
feature3, _ = np.histogram(cv.cvtColor(img, cv.COLOR_BGR2HSV)[:, :, 0].ravel(), bins=16)
feature4 = np.hstack([feature1, feature2, feature3])
# plt.plot(feature4)


# 方式三：HOG(自定义方法)
def hog(img):
    bin_n = 16
    h, w = img.shape
    # 计算梯度值
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    # 方向的划分
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16) 计算每个点属于那个方向
    # 每个cell的大小是10 * 10, 步长也是10
    hists = []
    for hi in range(h // 10):
        for wi in range(w // 10):
            bin_cell = bins[hi * 10:hi * 10 + 10, wi * 10:wi * 10 + 10]  # 当前这个cell各个像素点对应的方向
            mag_cell = mag[hi * 10:hi * 10 + 10, wi * 10:wi * 10 + 10]  # 当前这个cell各个相似度对应的梯度值
            hists.append(np.bincount(bin_cell.ravel(), mag_cell.ravel(), bin_n))  # 统计当前cell各个方向的梯度累加值
    # bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:] # cell的划分
    # mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:] # 对应的梯度大小
    # hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # 16 * h //10 * w // 10
    # plt.plot(hist)


# 方式三：HOG(方法调用)
# 像素大小（以像素为单位）（宽度，高度）。 它必须小于检测窗口的大小，
# 并且必须进行选择，以使生成的块大小小于检测窗口的大小。
img2 = cv.resize(img1, (510,600))
cell_size = (10,10)
# 每个方向（x, y）上每个块的单元数，必须选择为使结果块大小小于检测窗口
block = (2, 2)
# 块大小（以像素为单位）（宽度，高度）。必须是“单元格大小”的整数倍。
# 块大小必须小于检测窗口。
block_size = (block[0] * cell_size[0], block[1] * cell_size[1])
# 计算在x和y方向上适合我们图像的像素数
x_cells = img2.shape[1] // cell_size[0]
y_cells = img2.shape[0] // cell_size[1]
# 块之间的水平距离，以像元大小为单位。 必须为整数，并且必须
# 将其设置为（x_cells-block [0]）/ h_stride =整数。
h_stride = 1
# 块之间的垂直距离，以像元大小为单位。 必须为整数，并且必须
# 将其设置为 (y_cells - block[1]) / v_stride = integer.
v_stride = 1
# 块跨距（以像素为单位）（水平，垂直）。 必须是像素大小的整数倍。
block_stride = (cell_size[0] * h_stride, cell_size[1] * v_stride)
# 梯度定向箱的数量
num_bins = 9
# 指定检测窗口（感兴趣区域）的大小，以像素（宽度，高度）为单位。
# 它必须是“单元格大小”的整数倍，并且必须覆盖整个图像。
# 由于检测窗口必须是像元大小的整数倍，具体取决于您像元的大小，
# 因此生成的检测窗可能会比图像小一些。
# 完全可行
win_size = (x_cells * cell_size[0], y_cells * cell_size[1])
# https://blog.csdn.net/apple_52030329/article/details/131325512
hog1 = cv.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
hog_descriptor = hog1.compute(img2)

# 方式四：LBP：局部响应归一化，必须是灰度图
# 原始的LBP算子定义在像素3*3的邻域内，以邻域中心像素为阈值，相邻的8个像素的灰度值与邻域
# 中心的像素值进行比较，若周围像素大于中心像素值，则该像素点的位置被标记为1，否则为0。
# 这样，3*3邻域内的8个点经过比较可产生8位二进制数，将这8位二进制数依次排列形成一个
# 二进制数字，这个二进制数字就是中心像素的LBP值，
# https://zhuanlan.zhihu.com/p/556382573?utm_id=0
# https://www.lmlphp.com/user/60591/article/item/2496976/
from skimage.transform import rotate
from skimage.feature import local_binary_pattern, multiblock_lbp
from skimage import data, io,data_dir,filters, feature
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
radius = 1 # LBP算法中范围半径的取值
n_points = 8 * radius
lbp = local_binary_pattern(img2, n_points, radius)
mlbp = multiblock_lbp(img2, n_points, radius, width=16, height=16)

cv.imshow('lbp',lbp)
# cv.imshow('mlbp',mlbp)
cv.waitKey(0)
cv.destroyAllWindows()

plt.plot(hog_descriptor)
plt.show()

