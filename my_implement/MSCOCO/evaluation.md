# MSCOCO的评价指标

[参考1](https://www.bilibili.com/read/cv14176041), [参考2](https://blog.csdn.net/perfect_ch/article/details/117480528)。

将目标检测的边界框视为一个查询问题：
- 四类结果：
  - TP (True Positive): 真正例。Positive指预测输出为正，True代表预测正确
  - TN (True Negative): 真反例。预测输出为负，而且预测正确
  - FP (False Positive): 假正例。预测输出为正，但是预测错误
  - FN (False Negative): 假反例。预测输出为负，但是预测错误
- 评价：
  - 查准率precision：TP / (TP+FP)
  - 查全率recall：TP / (TP+FN)
- 对应到目标检测中：
  - 给定IoU_threshold，若预测框(True)和真实框的IOU大于阈值，则认为是positive的
  - AP: Precision-Recall曲线下方面积?
  - 算AP时，对每个类别分别计算，然后按类别做平均
  - AP：阈值从0.5到0.95，每隔0.05取一次值，算AP后做平均
  - $AP^{IoU=.50}$：阈值取0.5时的AP
  - $AP^{small}$：对小目标的AP


![指标](./figs/fdaef851b61fadc307283451473496c7dbadce61.png@942w_458h_progressive.webp)