---
layout:     post
title:      "instance segmentation之MNC"
subtitle:   "Instance-aware Semantic Segmentation via Multi-task Network Cascades"
date:       2018-06-05 01:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - instance segmentation
---

## 论文简介
* 地址：[Instance-aware Semantic Segmentation via Multi-task Network Cascades](https://arxiv.org/abs/1512.04412)
* 论文提出了一个可以用于多任务的级联网络，用于语义实例分割。模型主要有三个部分：区分实例、确定实例的mask以及确定实例的类别。级联的网络结构使得卷积特征可以被共享。
* 论文中的方法比当时其他具有类似精度的方法的速度(360ms)要快2个数量级。
* 论文中的模型也可以用于目标检测，精度超过了Faster-RCNN。
* 15年COCO语义分割的第一名。

## 主要部分
### 网络框架
* 主要是使用了级联的结构，实现了卷积特征参数的共享。

![MNC](/img/post/20180605-mnc框架.png)

### 论文流程
* Differentiating instances：给出所有实例的bounding box，但是这些实例的类别是未知的。
* Estimating masks：对实例的结果进行精细化，实现像素级的mask。
* Categorizing objects：判断出实例的类别。
* 为了实现反向传播，在标准的`max pooling`后面加一个可微的`warping layer`，来实现一个可微的`ROI pooling`。

### 损失函数
* 在阶段1中，网络结构与损失函数与RPN相同，输出为bounding box的信息以及它的概率p；阶段2中，对阶段1的结果做ROI pooling，再衔接2个FC层，第二个FC层的输出个数为$m^2$，与mask的大小相同(注意：论文中假定mask的大小是固定的)。对应会输出每个实例的mask；阶段3中，每个实例中只有mask部分对应的像素才会对损失函数做出贡献。
* 每个过程都包含一个损失函数，但是后一级的损失函数依赖于上一级的损失函数。
