import torch
from matplotlib import pyplot as plt


def bbox_corner_to_center(boxes:torch.Tensor):
    """从（左上，右下）转换到（中间，宽度，高度），boxes.shape[-1]==4"""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes_ = torch.stack((cx, cy, w, h), axis=-1)        # 第0维堆叠
    return boxes_

def bbox_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下），boxes.shape[-1]==4"""
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes_ = torch.stack((x1, y1, x2, y2), axis=-1)      # 第0维堆叠
    return boxes_

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


def IoU(bbox1, bbox2, is_corner=True):
    '''
    计算两个bbox的IoU，两个bbox均为corner格式

    两个框中，左上角点中靠右下角的，与右下角点中靠左上角的，形成的面积
    '''
    if not is_corner:
        bbox1_ = bbox_center_to_corner(bbox1)
        bbox2_ = bbox_center_to_corner(bbox2)
    else:
        bbox1_, bbox2_ = bbox1, bbox2
    
    in_h = min(bbox1_[3], bbox2_[3]) - max(bbox1_[1], bbox2_[1])
    in_w = min(bbox1_[2], bbox2_[2]) - max(bbox1_[0], bbox2_[0])
    intersection = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (bbox1_[2] - bbox1_[0]) * (bbox1_[3] - bbox1_[1]) + \
            (bbox2_[2] - bbox2_[0]) * (bbox2_[3] - bbox2_[1]) - intersection
    # print(intersection)
    iou = intersection / union
    return iou