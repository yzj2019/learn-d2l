import torch


def linreg(X, w, b):  #@save
    """线性回归模型"""
    return torch.matmul(X, w) + b

def MSELoss(y_hat, y):
    '''均方误差'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    '''小批量随机梯度下降'''
    with torch.no_grad():
        # 指明此步骤不参与梯度计算
        for param in params:
            param -= lr * param.grad / batch_size           # 按理来说，这里batch_size的归一化应该写在loss计算的时候进行
            param.grad.zero_()