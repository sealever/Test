from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.onnx import TrainingMode
from torch.utils.data import DataLoader, Dataset
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import onnxruntime


class IrisNet(nn.Module):
    def __init__(self):
        super(IrisNet, self).__init__()
        self.classify = nn.Sequential(
            nn.Linear(in_features=4, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=3)
        )

    def forward(self, x):
        return self.classify(x)


class NumpyDataset(Dataset):
    def __init__(self, x, y):
        super(NumpyDataset, self).__init__()
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def save(obj, path):
    torch.save(obj, path)


def load(net, path):
    print(f"模型恢复：{path}")
    ss_model = torch.load(path, map_location='cpu')
    net.load_state_dict(state_dict=ss_model['net'].state_dict(), strict=True)
    start_epoch = ss_model['epoch'] + 1
    best_acc = ss_model['acc']
    train_batch = ss_model['train_batch']
    test_batch = ss_model['test_batch']
    return start_epoch, best_acc, train_batch, test_batch


class Accuracy(nn.Module):
    def __init__(self):
        super(Accuracy, self).__init__()

    @torch.no_grad()
    def forward(self, scores, target) -> (int, torch.Tensor):
        """
        准确率计算方法
          N: 表示样本批次大小；
          C: 表示类别数目
        :param scores: 模型预测置信度对象 [N,C] float类型
        :param target: 样本实际标签类别对象 [N] long类型，内部就是[0,C)的索引id
        :return: (N,准确率值)
        """
        pre = torch.argmax(scores, dim=1)
        pre = pre.to(target.device, dtype=target.dtype)

        corr = (pre == target).to(dtype=torch.float)
        acc = torch.mean(corr)
        return corr.shape[0], acc


def training(batch_size):
    # 定义数据加载
    X, Y = load_iris(return_X_y=True)
    X = X.astype('float32')
    Y = Y.astype('int64')
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=1)

    # 构建Dataset对象
    train_data = NumpyDataset(x=x_train, y=y_train)
    test_data = NumpyDataset(x=x_test, y=y_test)

    # 构建数据遍历器
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=None
    )
    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size * 4,
        shuffle=True,
        num_workers=0,
        collate_fn=None
    )

    # 定义模型
    net = IrisNet()
    loss = nn.CrossEntropyLoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.01)

    # 模型恢复

    # 可视化输出
    now = datetime.now().strftime("%y%m%d")
    root_dir = Path(f'../output/03/{now}')
    summary_dir = root_dir / 'summary'
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True)
    checkout_dir = root_dir / 'model'
    if not checkout_dir.exists():
        checkout_dir.mkdir(parents=True)
    last_path = checkout_dir / 'last.pkl'
    best_path = checkout_dir / 'best.pkl'
    writer = SummaryWriter(log_dir=str(summary_dir))
    writer.add_graph(net, torch.rand(3, 4))

    # 遍历训练模型
    start_epoch = 0
    total_epoch = 100
    train_batch = 0
    test_batch = 0
    best_acc = 0

    # 模型恢复
    if best_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(net, best_path)
    elif last_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(net, best_path)

    for epoch in range(start_epoch, total_epoch + start_epoch):
        net.train()
        train_loss = []
        train_true, train_total = 0, 0
        for x, y in train_dataloader:
            # 前向过程
            scores = net(x)
            _loss = loss(scores, y)
            n, acc = acc_fn(scores, y)
            # 反向过程
            opt.zero_grad()
            _loss.backward()
            opt.step()

            _loss = _loss.item()
            acc = acc.item()
            train_total += n
            train_true += n * acc
            if train_batch % 2 == 0:
                print(f'epoch:{epoch},train batch:{train_batch}, loss:{_loss:.3f}, acc:{acc:.3f}')
                writer.add_scalar('train_loss', _loss, global_step=train_batch)
                writer.add_scalar('train_acc', acc, global_step=train_batch)
            train_batch += 1
            train_loss.append(_loss)

        # 评估
        net.eval()
        test_loss = []
        test_true, test_total = 0, 0
        with torch.no_grad():
            for x, y in test_dataloader:
                scores = net(x)
                _loss = loss(scores, y)
                n, acc = acc_fn(scores, y)

                _loss = _loss.item()
                acc = acc.item()
                test_total += n
                test_true += n * acc
                print(f'epoch:{epoch},test batch:{test_batch}, loss:{_loss:.3f}, acc:{acc:.3f}')
                writer.add_scalar('test_loss', _loss, global_step=test_batch)
                writer.add_scalar('test_acc', acc, global_step=test_batch)
                test_batch += 1
                test_loss.append(_loss)
        # epoch 阶段的信息可视化
        test_acc = test_true / test_total
        writer.add_scalars('loss', {'train': np.mean(train_loss), 'test': np.mean(test_loss)}, global_step=epoch)
        writer.add_scalars('acc', {'train': (train_true / train_total), 'test': (test_true / test_total)},
                           global_step=epoch)

        # 模型持久化
        if test_acc > best_acc:
            obj = {
                'net': net,
                'epoch': epoch,
                'train_batch': train_batch,
                'test_batch': test_batch,
                'acc': test_acc
            }
            save(obj, best_path.absolute())
            best_acc = test_acc
        # if epoch % 2 == 0:
        #     obj = {
        #         'net': net,
        #         'epoch': epoch,
        #         'train_batch': train_batch,
        #         'test_batch': test_batch,
        #         'acc': test_acc
        #     }
        #     save(obj, last_path.absolute())

    obj = {
        'net': net,
        'epoch': start_epoch + total_epoch - 1,
        'train_batch': train_batch,
        'test_batch': test_batch,
        'acc': test_acc
    }
    save(obj, last_path.absolute())
    writer.close()


def exports(model_dir):
    """
    NOTE: 可以通过netron（https://netron.app/）来查看网络结构
    将训练好的模型转换成可以支持多平台部署的结构，常用的结构：
    pt: Torch框架跨语言部署的结构
    onnx: 一种比较通用的深度学习模型框架结构
    tensorRT: 先转换成onnx，然后再进行转换使用TensorRT进行GPU加速
    openvino: 先转换为onnx，然后再进行转换使用OpenVINO进行GPU加速
    :param model_path:
    :return:
    """
    model_dir = Path(model_dir)

    # 模型恢复
    net = torch.load(model_dir / 'best.pkl', map_location='cpu')['net']
    net.eval().cpu()

    # 转换为pt结构（example为举例）
    example = torch.rand(1, 4)
    model1 = torch.jit.trace(net, example)
    model1.save(model_dir / 'best.pt')

    # 转换为onnx结构
    torch.onnx.export(
        model=net,  # 模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 选择训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes=None  # 给定是否是动态结构
    )
    torch.onnx.export(
        model=net,  # 模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best_dynamic.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 选择训练还是eval阶段
        input_names=['features'],  # 给定输入的tensor名称列表
        output_names=['label'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes={
            'features': {0: 'batch'},
            'label': {0: 'batch'}
        }  # 给定是否是动态结构
    )


@torch.no_grad()
def load_model(model_dir):
    model_dir = Path(model_dir)
    # pytorch 的模型恢复
    net1 = torch.load(model_dir / 'best.pkl', map_location='cpu')['net']
    net1.eval()
    # pytorch script 的模型恢复
    net2 = torch.jit.load(model_dir / 'best.pt', map_location='cpu')
    # onnx 的模型恢复
    net3 = onnxruntime.InferenceSession(str(model_dir / 'best_dynamic.onnx'))

    x = torch.rand(2, 4)
    print(net1(x))
    print(net2(x))
    print(net3.run(['label'], input_feed={'features': x.detach().numpy()}))


if __name__ == '__main__':
    # training(5)
    # exports(model_dir='../output/03/230818/model')
    load_model(model_dir='../output/03/230818/model')