from math import nan
import torch
import sys
sys.path.append('..')
import utils

def cross_entropy(y_hat, y):
    '''未解决溢出问题的交叉熵计算'''
    return -torch.log(y_hat[range(len(y_hat)), y])

def softmax(X):
    '''未解决溢出问题的softmax计算'''
    X_exp = torch.exp(X)    # 按元素
    partition = X_exp.sum(1, keepdim=True)  # 行和，压缩dim1
    return X_exp / partition    # 广播机制

def CrossEntropyLoss(O, y):
    '''解决溢出问题的CrossEntropyLoss，集成了softmax和cross_entropy，返回一维Tensor向量'''
    val, idx = O.max(dim=1, keepdim=True)
    O_sub = O - val                 # 减去每行的最大值
    O_exp = torch.exp(O_sub)        # 按元素做exp
    O_exp_sum = O_exp.sum(dim=1)    # 行和，已经压缩成Tensor向量了
    part2 = torch.log(O_exp_sum)    # 取自然对数
    part1 = O_sub[range(O_exp.shape[0]), y] # 类似上面的索引方式
    return - part1 + part2
    
def accuracy_ch3(y_hat, y):
    '''计算预测正确的数量，定义详见ch3'''
    if len(y_hat.shape)>1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluate_accuracy_ch3(net, data_iter):  #@save
    """计算在指定数据集上模型的预测准确率，定义详见ch3"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = utils.Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy_ch3(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_epoch_ch3(net, train_iter, loss, updater):  #@save
    """训练模型一个迭代周期（定义见第3章）"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = utils.Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            metric.add(float(l) * len(y), accuracy_ch3(y_hat, y), y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()              # loss返回的是一个大小为batch_size的向量
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy_ch3(y_hat, y), y.numel())
    # 返回本epoch的平均训练损失和平均训练精度
    return metric[0] / metric[2], metric[1] / metric[2]

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    animator = utils.Animator(xlabel='epoch', xlim=[1, num_epochs],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        # 逐个epoch迭代
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)     # 训练一个epoch
        test_acc = evaluate_accuracy_ch3(net, test_iter)                    # 评估测试集上的分类准确率
        animator.add(epoch + 1, train_metrics + (test_acc,))                # 这里的+是list合并，最后相当于[train_loss, train_acc, test_acc]
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, f'{train_loss}'
    assert train_acc <= 1 and train_acc > 0.7, f'{train_acc}'
    assert test_acc <= 1 and test_acc > 0.7, f'{test_acc}'

