import torch
from matplotlib import pyplot as plt


def bbox_corner_to_center(boxes:torch.Tensor):
    """从（左上，右下）转换到（中间，宽度，高度），boxes为n*4的张量或长度为4的向量"""
    sp = boxes.shape
    if len(sp)==1:
        boxes = boxes.view((1,4))
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)        # 第0维堆叠
    return boxes.view(sp)

def bbox_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下），boxes为n*4的张量或长度为4的向量"""
    sp = boxes.shape
    if len(sp)==1:
        boxes = boxes.view((1,4))
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)      # 第0维堆叠
    return boxes.view(sp)

def bbox_corner_to_rect(bbox, color):
    '''将单个边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式，bbox为向量'''
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)

def bbox_center_to_rect(bbox, color):
    '''将单个边界框(中心x,中心y,宽度,高度)格式转换成matplotlib格式，bbox为向量'''
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0] - 0.5 * bbox[2], bbox[1] - 0.5 * bbox[3]), width=bbox[2], height=bbox[3],
        fill=False, edgecolor=color, linewidth=2)


def show_bboxes(ax, bboxes, colors):
    '''
    添加颜色为colors的bboxes到坐标轴ax；

    bboxes和colors的格式暂定为list，bbox为corner表示
    '''
    assert len(bboxes)==len(colors)
    for bbox, color in zip(bboxes, colors):
        ax.add_patch(bbox_corner_to_rect(bbox, color))


def IoU(bbox1, bbox2):
    '''
    计算两个bbox的IoU，两个bbox均为corner格式

    两个框中，左上角点中靠右下角的，与右下角点中靠左上角的，形成的面积
    '''
    in_h = min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1])
    in_w = min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])
    inner = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + \
            (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - inner
    # print(inner)
    iou = inner / union
    return iou