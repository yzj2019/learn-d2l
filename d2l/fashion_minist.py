import torch
import torchvision
from torch.utils import data
from torchvision import transforms
import matplotlib.pyplot as plt


def load_data_fashion_mnist(batch_size:int, root:str, dataloader_workers:int =4, resize=None):  #@save
    """
    下载Fashion-MNIST数据集，然后将其加载到内存中
    ### Parameters:
    - batch_size: 批量的大小
    - root: 下载后存放数据集的位置
    - dataloader_workers: 取batch的线程数
    - resize: 如果需要transforms.Resize操作，则传入resize后的大小；比如想要将28*28的转变为64*64，则指定resize=64
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=dataloader_workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=dataloader_workers))

def get_fashion_mnist_labels(labels):  #@save
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes