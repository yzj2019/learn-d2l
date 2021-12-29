import torch

def pool2d(X, pool_size, mode='max'):
    '''2d单通道无padding无stride的池化操作，pool_size = (p_h, p_w)，mode = 'max' or 'avg' or 'min' '''
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i:i+p_h, j:j+p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i:i+p_h, j:j+p_w].mean()
            elif mode == 'min':
                Y[i, j] = X[i:i+p_h, j:j+p_w].min()
    return Y

def pool2d_multi_in_out(X, pool_size, mode='max'):
    '''2d多通道无padding无stride的池化操作，pool_size = (p_h, p_w)，mode = 'max' or 'avg' or 'min' '''
    return torch.stack([pool2d(x, pool_size, mode) for x in X], 0)   # list->第0维压缩成张量