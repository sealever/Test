import cv2 as cv
import numpy as np

# CV基本绘图
# img：给定绘画的对象
# color：给定像素点的颜色
# thickness：给定线条粗细程度,-1表示填充图像
# lineType：给定线条的类型
img = np.zeros((512,512,3), np.uint8)
# 画线
cv.line(img, pt1=(0,0), pt2=(511,511), color=(255,0,0), thickness=2)
# 画矩形
cv.rectangle(img, pt1=(10,10), pt2=(50,320), color=(0,255,0),thickness=-1)
# 画圆
cv.circle(img, center=(200,200), radius=100, color=(0,0,255), thickness=10)
# 画椭圆
cv.ellipse(img, center=(350,350),axes=(100,50), angle=30, startAngle=0,endAngle=180, color=(0,90,255), thickness=-1)
# 画多边形
pts = np.array([[10,5], [20,50], [50,30], [80,85], [120,30]], np.int32)
pts = pts.reshape((-1, 1, 2))
print(pts.shape)
cv.polylines(img, [pts], isClosed=False, color=(255,46,23),thickness=4)
# 添加文本
font = cv.FONT_HERSHEY_TRIPLEX
cv.putText(img, text='cat', org=(400,50), fontFace=font, fontScale=2, color=(252,45,69), thickness=1, lineType=cv.LINE_4)
cv.rectangle(img, pt1=(400,50), pt2=(500,320), color=(0,255,0),thickness=2)

cv.imshow('image', img)
cv.waitKey(0)
cv.destroyAllWindows()

