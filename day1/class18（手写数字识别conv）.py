from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
from torch import optim
from torch.onnx import TrainingMode
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


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


class Network(nn.Module):
    def __init__(self, in_channels, num_class, img_high, img_width, units=None):
        super(Network, self).__init__()
        if units is None:
            units = [16, 'M', 32, 64, 'M', 64, 64]
        self.in_channels = in_channels
        self.num_class = num_class
        layers = []
        for unit in units:
            if unit == 'M':
                layers.append(nn.MaxPool2d(2, 2))
                img_high = int(np.floor((img_high - 2 + 2) / 2))
                img_width = int(np.floor((img_width - 2 + 2) / 2))
            else:
                layers.append(
                    nn.Conv2d(in_channels, unit, kernel_size=(3, 3), stride=(1, 1), padding=1)
                )
                layers.append(nn.ReLU())
                in_channels = unit
        layers.append(nn.Flatten(1))
        layers.append(nn.Linear(in_features=in_channels * img_width * img_high, out_features=num_class))
        self.classify = nn.Sequential(*layers)

    def forward(self, x):
        """
        手写数字识别的前向执行过程
        :param x: [N,C,H,W] 也就是 [N,1,28,28]表示N个手写数字图像的原始特征数据
        :return: [N,3] scores N表示N个样本，3表示3个类别，每个样本属于每个类别的置信度值
        """

        return self.classify(x)


def save(obj, path):
    torch.save(obj, path)


def load(path, net):
    print(f'模型恢复：{path}')
    model = torch.load(path, map_location='cpu')
    net.load_state_dict(state_dict=model['net'].state_dict(), strict=True)
    start_epoch = model['epoch'] + 1
    best_acc = model['acc']
    train_batch = model['train_batch']
    test_batch = model['test_batch']
    return start_epoch, best_acc, train_batch, test_batch


def training():
    # now = datetime.now().strftime('%y%m%d')
    now = '230727212952'
    root_dir = Path(f'../output/05/{now}')
    summary_dir = root_dir / 'summary'
    if not summary_dir.exists():
        summary_dir.mkdir(parents=True)
    checkout_dir = root_dir / 'model'
    if not checkout_dir.exists():
        checkout_dir.mkdir(parents=True)
    last_path = checkout_dir / 'last.pkl'
    best_path = checkout_dir / 'best.pkl'
    total_epoch = 100
    summary_interval_batch = 2
    save_interval_batch = 2
    start_epoch = 0
    best_acc = -1.0
    train_batch = 0
    test_batch = 0
    batch_size = 16

    # 1.定义数据加载
    train_dataset = datasets.MNIST(
        root='../datas/MNIST',
        train=True,
        transform=transforms.ToTensor(),
        download=True
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataset = datasets.MNIST(
        root='../datas/MNIST',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size * 2)

    # 2.定义模型
    net = Network(in_channels=1, num_class=10, img_high=28, img_width=28, units=None)
    loss_fn = nn.CrossEntropyLoss()
    acc_fn = Accuracy()
    opt = optim.SGD(params=net.parameters(), lr=0.01)

    # 3.模型恢复
    if best_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(best_path, net)
    elif last_path.exists():
        start_epoch, best_acc, train_batch, test_batch = load(best_path, net)

    # 4.定义可视化输出
    writer = SummaryWriter(log_dir=str(summary_dir))
    writer.add_graph(net, torch.rand(3, 1, 28, 28))

    # 5.遍历训练模型
    for epoch in range(start_epoch, total_epoch + start_epoch):
        # 5.1 训练
        net.train()
        train_losses = []
        train_true, train_total = 0, 0
        for batch_img, batch_label in train_dataloader:
            # 前向过程
            scores = net(batch_img)  # [N,10] 得到的是每个样本属于各个类别的置信度
            loss = loss_fn(scores, batch_label)
            n, acc = acc_fn(scores, batch_label)
            # 反向过程
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss = loss.item()
            acc = acc.item()
            train_total += n
            train_true += n * acc
            if train_batch % summary_interval_batch == 0:
                print(f'epoch:{epoch}, train batch:{train_batch}, loss:{loss:.3f}, acc:{acc:.3f}')
                writer.add_scalar('train_loss', loss, global_step=train_batch)
                writer.add_scalar('train_acc', acc, global_step=train_batch)
            train_batch += 1
            train_losses.append(loss)

        # 5.2 评估
        net.eval()
        test_losses = []
        test_true, test_total = 0, 0
        with torch.no_grad():
            for batch_img, batch_label in test_dataloader:
                # 前向过程
                scores = net(batch_img)
                loss = loss_fn(scores, batch_label)
                n, acc = acc_fn(scores, batch_label)

                loss = loss.item()
                acc = acc.item()
                test_total += n
                test_true += n * acc
                print(f'epoch:{epoch}, test batch:{test_batch}, loss:{loss:.3f}, acc:{acc:.3f}')
                writer.add_scalar('test_loss', loss, global_step=test_batch)
                writer.add_scalar('test_acc', acc, global_step=test_batch)
                test_batch += 1
                test_losses.append(loss)

        # 5.3 epoch阶段的信息可视化
        train_losses = np.mean(train_losses)
        test_loss = np.mean(test_losses)
        train_acc = train_true / train_total
        test_acc = test_true / test_total
        writer.add_scalars('loss', {'train': train_losses, 'test': test_loss}, global_step=epoch)
        writer.add_scalars('acc', {'train': train_acc, 'test': test_acc}, global_step=epoch)

        # 5.4 模型持久化
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
        if epoch % save_interval_batch == 0:
            obj = {
                'net': net,
                'epoch': epoch,
                'train_batch': train_batch,
                'test_batch': test_batch,
                'acc': test_acc
            }
            save(obj, last_path.absolute())

    # 6. 最终模型持久化
    obj = {
        'net': net,
        'epoch': start_epoch + total_epoch - 1,
        'train_batch': train_batch,
        'test_batch': test_batch,
        'acc': test_acc
    }
    save(obj, last_path.absolute())
    writer.close()


def export(model_dir):
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

    # 模型转换为pt结构
    example = torch.rand(1, 1, 28, 28)
    module = torch.jit.trace(net, example)
    module.save(model_dir / 'best.pt')

    # 转换为onnx结构
    torch.onnx.export(
        model=net,
        args=example,
        f=model_dir / 'best.onnx',
        training=TrainingMode.EVAL,
        input_names=['images'],
        output_names=['scores'],
        opset_version=12,
        dynamic_axes=None
    )
    torch.onnx.export(
        model=net,  # 给定模型对象
        args=example,  # 给定模型forward的输出参数
        f=model_dir / 'best_dynamic.onnx',  # 输出文件名称
        training=TrainingMode.EVAL,  # 训练还是eval阶段
        input_names=['images'],  # 给定输入的tensor名称列表
        output_names=['scores'],  # 给定输出的tensor名称列表
        opset_version=12,
        dynamic_axes={
            'images': {
                0: 'batch'
            },
            'scores': {
                0: 'batch'
            }
        }  # 给定是否是动态结构
    )


@torch.no_grad()
def load_model(model_dir):
    model_dir = Path(model_dir)
    net1 = torch.load(model_dir / 'best.pkl', map_location='cpu')['net']
    net1.eval().cpu()
    net2 = torch.jit.load(model_dir / 'best.pt', map_location='cpu')
    net2.eval().cpu()
    net3 = onnxruntime.InferenceSession(str(model_dir / 'best_dynamic.onnx'))
    x = torch.rand(2, 1, 28, 28)
    print(net1(x))
    print(net2(x))
    print(net3.run(['scores'], input_feed={'images': x.detach().numpy()}))

    img_paths = [
        r"..\datas\MNIST\MNIST\images\0\63.png",
        r"..\datas\MNIST\MNIST\images\1\77.png",
        r"..\datas\MNIST\MNIST\images\2\82.png",
        r"..\datas\MNIST\MNIST\images\3\12.png",
        r"..\datas\MNIST\MNIST\images\4\26.png",
        r"..\datas\MNIST\MNIST\images\5\35.png",
        r"..\datas\MNIST\MNIST\images\6\36.png",
        r"..\datas\MNIST\MNIST\images\7\52.png",
        r"..\datas\MNIST\MNIST\images\8\55.png",
        r"..\datas\MNIST\MNIST\images\9\54.png",
    ]
    for img_path in img_paths:
        img = plt.imread(img_path)[:, :, 0][None, None, :, :]
        img = torch.from_numpy(img)
        print('=' * 100)
        print(img_path)
        print(torch.argmax(net1(img), dim=1))
        print(net2(img))
        print(net3.run(['scores'], input_feed={'images': img.detach().numpy()}))


if __name__ == '__main__':
    training()
    # export(model_dir='../output/05/230727212952/model')
    # load_model(model_dir='../output/05/230727212952/model')
