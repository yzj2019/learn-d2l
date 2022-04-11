import torch

def try_gpu(i=0):
    '''如果存在，则返回GPU(i)，否则返回CPU()'''
    if torch.cuda.device_count() > i:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():
    '''返回所有可用的GPU，如果没有GPU，则返回[cpu(), ]'''
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

if __name__=='__main__':
    try_gpu(0), try_gpu(10), try_all_gpus()