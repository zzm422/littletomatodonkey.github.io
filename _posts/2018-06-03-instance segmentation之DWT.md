---
layout:     post
title:      "instance segmentation之DWT"
subtitle:   "deep-watershed transform"
date:       2018-06-03 10:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - instance segmentation
---

## 论文简介
* 地址：[Deep Watershed Transform for Instance Segmentation](https://arxiv.org/abs/1611.08303)
* 论文主要是使用了DNN的思想，实现end-to-end的实例分割

## 关于实例分割的研究进展
* 传统的分水岭算法容易造成过分割。改进的分水岭算法有
    * 首先预估instance的位置，然后再确定basin。
    * 启发式优化算法对分水岭算法的basin的相对深度进行估计，但是模型精度较差。
上面2种方法实现起来都比较困难。
* 基于候选区域的refinement
* 深度结构化模型：结合DNN与CRF等
模板匹配：使用CNN提取图像特征，对一个instance中的每个pixel赋予label
* RNN：记录上一帧instance分割结果，用于预测下一帧的instance分割。
* 使用CNN与通用的聚类算法，直接给出instance的数目与bounding box，同时给出每个pixel的置信度得分。
* 递归候选区域

## 主要优点
* 直接学习分水岭变换的能量，每个basin都对应一个instance，同时分割脊在能量域中的高度都相同。
* 主要使用了end-to-end的深度分水岭算法，模型精度比state-of-art好很多。
* 分割结果与instance的个数无关，这与一些RNN方法不同。

## 流程
* 模型将RGB原图与语义分割结果作为输入，相当于是一个4-channel的图像，本文中使用了PSPNet的语义分割结果。
* 过滤掉语义分割中的背景部分。
* 对语义分割结果的label进行缩放，使其成为等间隔的。
* 构建Direction Network，输出为每个像素的能量梯度(x&y,共2通道)，在这里面，使用VGG-Net进行特征提取，将MSE作为DN的损失函数。
* 构建Watershed Transform Network(WTN)，损失函数是修正的交叉熵函数。
* 构建级联网络，对网络进行fine-tune，将groud truth的距离变换作为训练目标。
* 后处理：对instance分割结果进行膨胀等结构化的处理，去除一些面积很小的instance。