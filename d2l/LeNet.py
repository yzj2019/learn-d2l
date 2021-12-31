import torch
from torch import nn
import sys
from d2l.Softmax import accuracy_ch3
sys.path.append('..')
import utils

def evaluate_accuracy_gpu(net, data_iter, device=None):
    '''使用GPU计算模型在数据集上的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device    # 不指定计算设备，则使用模型所在的设备
    metric = utils.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):
            X = [x.to(device) for x in X]
        else:
            X = X.to(device)
        y = y.to(device)
        metric.add(accuracy_ch3(net(X), y), y.numel())
    return metric[0] / metric[1]


#@save
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = utils.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = utils.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，范例数
        metric = utils.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy_ch3(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


class LeNet_5(nn.Module):
    '''LeNet-5的pytorch实现，去掉了最后的高斯激活'''
    def __init__(self):
        # 1.调用父类的初始化
        super(LeNet_5, self).__init__()
        # 2.定义我们需要哪些函数
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.avgpool2d = nn.AvgPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten()     # 展平dim[1:]
        self.linear1 = nn.Linear(16*5*5, 120)
        self.Linear2 = nn.Linear(120, 84)
        self.Linear3 = nn.Linear(84, 10)
    def forward(self, X):
        '''定义前向计算过程'''
        # 卷积核1
        y = self.conv1(X.view(-1, 1, 28, 28))
        y = self.sigmoid(y)
        y = self.avgpool2d(y)
        # 卷积核2
        y = self.conv2(y)
        y = self.sigmoid(y)
        y = self.avgpool2d(y)
        # 展平、线性层
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.sigmoid(y)
        y = self.Linear2(y)
        y = self.sigmoid(y)
        y = self.Linear3(y)
        return y