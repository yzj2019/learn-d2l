import torch
from torch.utils import data

def synthetic_data(w, b, num_examples):
    '''生成 y = Xw + b + 高斯噪声'''
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    # print(y.shape)
    y += torch.normal(0, 0.01, y.shape)
    # print(y.shape)
    return X, y.reshape((-1, 1))                # matmul后，降维了，需要reshape成二维的，方便后面算loss

def load_array(data_arrays, batch_size, is_train=True):
    '''利用data.TensorDataset构建dataset，is_train代表是否为训练集，是的话需要shuffle'''
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    '''平方误差'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():
        # 指明此步骤不参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size           # 按理来说，这里batch_size的归一化应该写在loss计算的时候进行
            param.grad.zero_()