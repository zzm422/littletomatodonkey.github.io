---
layout:     post
title:      "2018-06-03-instance segmentation之无proposal的聚类"
subtitle:   "Cluster for Proposal-Free"
date:       2018-06-03 20:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - instance segmentation
---

## 论文简介
* 地址：[Learning to Cluster for Proposal-Free Instance Segmentation](https://arxiv.org/abs/1803.06459)
* 论文使用DNN，提出了一种新的end-to-end的图像像素聚类方法。使用`instance labeling`中最基本的属性：pixel之间的成对关系，来训练网络，最后使用`图着色理论`来对目标进行着色，论文中的模型获得了`2017 CVPR Autonomous Driving Challenge`的第二名。
* 论文最主要的贡献有
    * 提出一种新的目标函数，来训练用于实例分割的FCN
    * 结合图的着色理论，讲述了一种对目标进行augment的方法，即对于没有相连的实例，他们可以使用相同的颜色来显示。
    * 从经验上验证了FCN可以用于解决end-to-end的图像像素聚类问题，之前的很多方法大多是基于proposal的方法或者是需要借助语义分割等中间过程的结果进行实例分割。


## 论文方法

### Learning Instance Labeling
* 模型的输入为RGB图像，输出为基于每个实例的mask，每个实例都有唯一的id。这里的id并不是唯一的，即2个实例交换id，得到的结果也是一个有效的分割。

#### Learning Objective
* FCN的输出为pixel属于特定instance的概率，这符合多项式分布。有以下假设：如果两个pixel属于同一个instance，则它们的分布是相似的，否则是不相似的。loss function是关于像素对的hinge-loss，具体表达式见原文。

#### 采样策略
* 因为像素对的个数与像素点的个数平方成正比，因此有必要对图像中的像素进行采样。在训练阶段，每次都会采样固定数目的像素点(N,比如1000个)，只选择那些有属于特定instance的像素点。最终得到的像素对有$N^2$个。
* 把背景作为单独的instance(也会被采样)，每个instance采样的像素点个数是相同的，与他们的面积无关，因此背景pixel太多，对背景采样得到的像素点是十分稀疏的。
* 增加对背景的采样像素点的个数就可以提升模型的精度，背景像素点的loss是二分类的loss(类似于logostic回归的loss)，非背景则采用论文中提到的`pairwise loss`。

#### 处理无限数目的instance的方法
* 如果处理无限数目的instance，则会有2个问题，一方面一张图像中的instance数目基本符合`长尾分布`，大部分图像中的instance数目都不会很多，因此对应的训练数据很少，同时更高维的输出矩阵也会增加模型的计算时间。
* 基于上面的问题，论文借鉴图着色的方法，对instance的index进行重新指派，从而避免了instance数目过多的问题。

### 其他的一些细节
* 当instance数目是有限的时候，可以直接实现end-to-end的实例分割，不需要后处理的操作；而对于无限数目的instance的解决方法，我们需要使用连通区域分析进行后处理，得到最终的结果。
* 论文中的模型是不知道instance所属的类别的，因此需要结合语义分割。对于特定的instance，对对应的那些mask位置的语义分割结果求平均，最终得到instance所属的类别。