---
layout:     post
title:      "instance segmentation之车道线检测"
subtitle:   "Towards End-to-End Lane Detection: an Instance Segmentation Approach"
date:       2018-06-04 10:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - instance segmentation
    - autonomous driving
---

## 论文简介
* 地址：[Towards End-to-End Lane Detection: an Instance Segmentation Approach](https://arxiv.org/abs/1802.05591)
* 传统的车道线检测方法十分依赖人工选取特征，需要大量的后处理步骤，对于已经有的一些DNN的方法，它们只能处理预先定义好的、数目固定的车道线，无法处理车道线发生变化的情况。
* 引入instance segmentation的方法，将每个车道线都视为一个instance，论文中提出一种使用学习得到的透视变换方法，保证模型在路面发生变化时也会有较强的鲁棒性。
* 论文提出的方法可以处理任意车道线数目、车道线变化的情况，同时能够做到实时(50fps,图像大小为512X256,Nvidia 1080Ti)。

## 论文主要的组成部分
### LaneNet
* 网络结构如下图。使用一个共享的encoder，对输入图像进行处理，得到2个branch：嵌入branch和语义分割的branch。嵌入branch可以将不同的车道线区分为不同的instance；因为只需要考虑车道线，因此语义分割的结果是二值化图像；然后对2个branch做聚类，最终得到结果。

![LaneNet](/img/post/20180602-lanenet.png)

### 基于学习方法的投影变换方法()H-Net
* 将输入的RGB图像作为输入，使用LaneNet得到输出的实例分割结果，然后将车道线像素使用H-Net输出得到的透视变换矩阵进行变换，对变换后的车道线像素在变化后的空间中进行拟合，再将拟合结果经过逆投影变换，最终得到原始视野中的车道线拟合结果。
* H-Net将RGB作为输入，输出为基于该图像的透视变换系数矩阵，优化目标为车道线拟合效果。

### 关于嵌入
* 嵌入的方法与[https://arxiv.org/abs/1708.02551](https://arxiv.org/abs/1708.02551)中介绍的类似，只是修改了hinge loss的阈值，设置为$\delta _v > 6\delta _d$。


