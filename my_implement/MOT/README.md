参考[PaddleDection](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/configs/mot/README_cn.md)

当前主流的多目标追踪(MOT)算法主要由两部分组成：Detection+Embedding。Detection部分即针对视频，检测出每一帧中的潜在目标。Embedding部分则将检出的目标分配和更新到已有的对应轨迹上(即ReID重识别任务)。根据这两部分实现的不同，又可以划分为SDE系列和JDE系列算法。

SDE(Separate Detection and Embedding)这类算法完全分离Detection和Embedding两个环节，最具代表性的就是[SORT](https://github.com/abewley/sort)和[DeepSORT](https://github.com/nwojke/deep_sort)算法。这样的设计可以使系统无差别的适配各类检测器，可以针对两个部分分别调优，但由于流程上是串联的导致速度慢耗时较长，在构建实时MOT系统中面临较大挑战。

JDE(Joint Detection and Embedding)这类算法完是在一个共享神经网络中同时学习Detection和Embedding，使用一个多任务学习的思路设置损失函数。代表性的算法有JDE（可参考飞桨的那份实现）和FairMOT。这样的设计兼顾精度和速度，可以实现高精度的实时多目标跟踪。