import pickle
from glob import glob
from pathlib import Path

import numpy as np
import cv2 as cv
from skimage.segmentation import felzenszwalb
from tqdm import tqdm


# 1. 针对原始图像进行区域的划分，选除候选框 --> 区域提名
def stage1():
    out_dir = Path('../output/01/dog/images')
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv.imread(r'G:\AI-study\class\datas\MNIST\tietu.png')
    h, w, _ = img.shape
    scales = [
        # [400, 450],
        [200, 200]
        # [20, 20]
    ]
    rk = 80
    for kh, kw in tqdm(scales):
        for sh in range(0, h-kh, rk):
            eh = sh + kh
            for sw in range(0, w-kw, rk):
                ew = sw + kw
                cv.imwrite(str(out_dir / f"{kh}_{kw}_{sh}_{sw}.png"), img[sh:eh, sw:ew, :])

def stage1_felzenszwalb():
    out_dir = Path('../output/01/dog/images')
    out_dir.mkdir(parents=True, exist_ok=True)
    img = cv.imread(r'G:\AI-study\class\datas\MNIST\tietu.png')
    img_mask = felzenszwalb(img)
    cv.imshow('img', img)
    cv.imshow('img_mask', img_mask / img_mask.max())
    cv.waitKey(0)
    cv.destroyAllWindows()



# 2. 针对每个候选框提取出对应的特征信息
def stage2():
    def _img_feature(_img):
        # 转换图像特征
        # g_feature = np.histogram(_img[:,:,0].ravel(), bins=25)
        # b_feature = np.histogram(_img[:,:,1].ravel(), bins=25)
        # r_feature = np.histogram(_img[:,:,2].ravel(), bins=25)
        # c = np.hstack([g_feature, b_feature, r_feature])
        return np.bincount(_img.ravel(), minlength=256)
    img_files = glob('../output/01/dog/images/*.png')
    out_dir = Path("../output/01/dog")
    out_dir.mkdir(parents=True, exist_ok=True)
    features = []
    for img_file in tqdm(img_files):
        img = cv.imread(img_file)
        features.append(_img_feature(img))
    features = np.asarray(features)
    with open(str(out_dir / "feature.pkl"), 'wb') as writer:
        pickle.dump({'files': img_files, 'features': features}, writer)


# 3. 针对每个候选区域使用模型进行预测，选择出预测概率最大的候选框作为预测边框即可
# NOTE: 模型训练是一个独立的图像分类模型 --> 需要先独立训练一个图像分类模型(类别:背景、物体1、物体2....，模型可以是：svg、gbdt.....)
def stage3():
    out_dir = Path("../output/01/dog")
    with open(str(out_dir / "feature.pkl"), 'rb') as reader:
        objs = pickle.load(reader)
        img_files = objs['files']
        feature = objs['features']
    # # 恢复模型
    # model = joblib.load("xxx")
    #
    # # 对数据进行预测 --> 属于各个类别的概率值
    # scores = model.predict(features) # [N,num_classes]
    #
    # # 针对每个类别排序，选择概率最大的对应类别下标 --> 最大的概率对应的边框位置(img_files中获取)


if __name__ == '__main__':
    # stage1()
    # stage1_felzenszwalb()
    stage2()
    # stage3()
