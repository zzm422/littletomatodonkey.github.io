---
layout:     post
title:      "instance segmentation之PANet"
subtitle:   "Path Aggregation Network for Instance Segmentation"
date:       2018-06-05 08:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - instance segmentation
---

## 论文简介
* 地址：[Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)
* 论文提出的PANet是一个基于proposal的实例分割框架，缩短了底层和高层特征的通路；提出了自适应的特征pooling，可以各个各个尺度中的特征；在预测mask的时候，增加一个额外的FC旁路，辅助FCN分割支路的结果。
* COCO 17实例分割第一名，检测任务第二名。

## 主要部分
* 论文模型框架如下图所示

![MNC](/img/post/20180605-PANet框架.png)

### Bottom-up Path Augmentation
* 基于FPN网络，PANet新构建了N2~N5，N2与P2完全相同，N3~N5与P3~P5的尺寸是相同的，$N_i$得到的输出与$P_{i+1}$逐像素叠加，新建的之路缩短了底层尺寸的大的特征到高层尺寸小的特征之间的距离，实现更加有效的特征融合。

### Adaptive Feature Pooling
* 高层特征由大的感受野生成，可以捕捉更加丰富的上下文信息；而低层特征可以获取更多的细节以及更高的定位精度。基于这样的结论，针对每个proposal，PANet对其在所有层的特征进行池化，并通过FC层，最后进行融合，在通过FC层，得到最终的分割结果。因为是对所有层的特征进行池化并融合，作者将其称之为`自适应特征池化`。

### Mask Prediction Structure
* FCN(全卷积神经网络)基于局部感受野的特征和参数给出每个像素的预测结果，特征参数在不同的参数都是共享的，它对空间信息不敏感。而fc(全连接层)对空间位置很敏感，因为它在不同的空间位置使用不同的参数对像素进行预测。因此考虑将2种方法进行融合。
* 在`Mask R-CNN`的基础上增加了一个fc层，对fc的结果进行reshape，然后将其与`Mask R-CNN`的输出结果进行融合，得到最终的mask。