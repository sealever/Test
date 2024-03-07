import numpy as np
from iris_processor import IrisProcessor

processor = IrisProcessor(r'G:\AI-study\class\output\03\230818\model\best_dynamic.onnx')

while True:
    x = input('请输入特征属性，使用空格隔开:')
    if 'q' == x:
        break
    x = x.split(" ")
    x = [int(i) for i in x]
    if len(x) != 4:
        print(f"输入特征属性异常，请输入4维特征属性:{x}")
        continue
    print(type(x))
    x = np.asarray([x])
    r = processor.predict(x)
    print(f"预测结果为:{r}")
