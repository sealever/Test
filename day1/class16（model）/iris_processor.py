import os

import numpy as np
import onnxruntime
import torch


def softmax(scores):
    """
    求解softmax概率值
    :param scores: numpy对象 [n,m]
    :return: 求解属于m个类别的概率值
    """
    a = np.exp(scores)
    b = np.sum(a, axis=1, keepdims=True)
    p = a / b
    return p


class IrisProcessor(object):
    def __init__(self, model_path):
        super(IrisProcessor, self).__init__()
        model_path = os.path.abspath(model_path)
        _, ext = os.path.splitext(model_path.lower())
        self.pt, self.onnx = False, False
        if ext == '.pt':
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval().cpu()
            self.model = model
            self.pt = True
        elif ext == '.onnx':
            session = onnxruntime.InferenceSession(model_path)
            self.session = session
            self.input_name = 'features'
            self.output_name = 'label'
            self.onnx = True
        else:
            raise ValueError(f'当前仅支持pt和onnx格式，当前文件类型为：{model_path}')
        self.classes = ['猫', '狗', '羊']
        print(f'模型恢复成功：pt->{self.pt}; onnx->{self.onnx}')

    def after_model(self, x, scores):
        probas = softmax(scores)
        index = np.argmax(scores, axis=1)
        result = []
        for k, idx in enumerate(index):
            r = {
                'id': int(idx),
                'label': self.classes[idx],
                'proba': float(probas[k][idx])
            }
            result.append(r)
        return result

    @torch.no_grad()
    def predict_pt(self, x):
        tensor_x = torch.from_numpy(x).to(torch.float)
        scores = self.model(tensor_x)  # [n,4] -> [n,3]
        scores = scores.numpy()  # tensor -> numpy
        return self.after_model(x, scores)

    def predict_onnx(self, x):
        onnx_x = x.astype('float32')
        # session.run会返回output_names给定的每个名称对应的预测结果，最终是一个list列表，列表大小和参数output_names大小一致
        scores = self.session.run(
            output_names=[self.output_name],
            input_feed={self.input_name:onnx_x}
        )  # [n,4] -> list([n,3])
        scores = scores[0]  # 获取第一个输出结果（output_name对应结果）
        return self.after_model(x, scores)

    def predict(self, x):
        """
        模型预测方法，输入鸢尾花的原始特征属性，返回对应的预测标签
        :param x: numpy对象，形状为[n,4]表示n个样本，4个属性
        :return: 每个样本均返回对应的预测类别名称、id以及概率值，以dict形式返回
        """
        if self.pt:
            return self.predict_pt(x)
        elif self.onnx:
            return self.predict_onnx(x)
        else:
            raise ValueError("当前模型初始化异常!")
