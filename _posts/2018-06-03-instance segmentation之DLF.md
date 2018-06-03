---
layout:     post
title:      "instance segmentation之DLF"
subtitle:   "Discriminative Loss Function"
date:       2018-06-03 10:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - instance segmentation
---

## 论文简介
* 地址：[Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551)
* 代码：[https://github.com/DavyNeven/fastSceneUnderstanding](https://github.com/DavyNeven/fastSceneUnderstanding)
* 引入了判别式损失函数，前端网络可以使用之前已经有的网络结构等，然后将损失函数修改为DLF，再进行重新训练。
* 论文中的模型不依赖物体的proposal或者是其他的一些循环机制等。论文的主要贡献验证了即使不使用很复杂的方法，论文中的模型也可以与更加复杂的复杂的方法媲美。
* 提出的模型不对语义分割与实例分割做特别的区分，对于这两种任务所使用的网络结构都是相同的。

## Discriminative loss function
* 像素嵌入(pixel embedding)指的是将每个pixel映射到n维的特征空间中，label相同的pixel在映射空间中距离很近，而不同的pixel会相互远离。
* 定义一个cluster为共享同一label的像素嵌入。DLF包含了cluster之间与内部的pull与push force。DLF包含三项：
    * 方差项，即嵌入空间中，cluster内部pull force：获取一个实例的所有像素，然后计算平均值，将属于这个实例的所有的像素点都向一个点拉，从而减小实例的嵌入方差。
    * 距离项：即cluster之间的push force：获取所有cluster的中心点，这种推力会将它们推得更远。
    * 正则化：小的拉力项，使得所有的cluster离原点不至于太远。

具体的公式见论文原文，最终的DLF是上述三项的线性加权。
* 在计算loss的时候，有2个阈值变量
    * $\delta _v$：在cluster内部，距离中心小于这个阈值的嵌入的向量都不会产生loss，相当于hinge正则化，这也会使得特征空间不会收敛到一个点。
    * $\delta _d$：距离大于$2\delta _d$的类簇中心之间不会有推力，在没有其他力的情况下，可以自由移动。


## 一些其他的说明与处理
* 需要对模型输出做后处理(基于损失函数的阈值操作)
* 模型对于有重叠的instance segmentation的效果也很好
* 如果增加需要分割的类，则需要重新设计网络架构并训练，因为输出的特征空间的维度与类别个数是相同的，这类似于softmax。
* 有以下结论
    * 所有的嵌入距离他们的簇中心的距离都不大于$\delta _v$。
    * 任意两个簇中心的距离都不小于$2\delta _d$.
    * 论文中设置$\delta _v > 2\delta _d$，能够保证所有簇中的嵌入都比到簇外的嵌入的距离要小。
* 如果使用语义分割的groundtruth，而非语义分割的预测结果，则模型的精度会大幅提升。