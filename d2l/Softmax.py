import torch

def cross_entropy(y_hat, y):
    '''交叉熵计算'''
    return -torch.log(y_hat[range(len(y_hat)), y])

def softmax(X):
    '''未解决溢出问题的softmax计算'''
    X_exp = torch.exp(X)    # 按元素
    partition = X_exp.sum(1, keepdim=True)  # 行和，压缩dim1
    return X_exp / partition    # 广播机制
    
