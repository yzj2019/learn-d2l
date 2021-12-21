import torch

def corr2d(X, K):
    '''二维互相关运算，X、K为2d'''
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i+h, j:j+w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    '''多通道输入的2d互相关运算，X为3d，K为3d'''
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    '''多输入多输出通道的2d互相关运算，X为3d，K为4d'''
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)   # list->第0维压缩成张量

def corr2d_multi_in_out_1x1(X, K):
    '''多输入多输出通道的1*1卷积，使用全连接方式实现'''
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h*w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))