---
layout:     post
title:      "MOT之instance flow"
subtitle:   "INSTANCE FLOW BASED ONLINE MULTIPLE OBJECT TRACKING"
date:       2018-06-05 12:00:00
author:     "donkey"
header-img: "img/post/blog-2018.jpg"
catalog: true
tags:
    - 论文
    - MOT
---

## 论文简介
* 地址：[INSTANCE FLOW BASED ONLINE MULTIPLE OBJECT TRACKING](https://arxiv.org/abs/1703.01289)
* 论文中实现了单目视频数据中，对于已知类别的物体进行在线多目标跟踪的方法。一般的多目标跟踪都是基于bouding box的，即输出的结果为关于物体的bounding box或者附加其置信度，而本文的结果是基于实例分割的结果，最终给出的结果是跟踪对象的mask，实现了像素级别的跟踪。
* 论文基于光流法，预测实例在下一帧的位置与形状；定义相似度矩阵，用于判断相邻两帧之间的实例的相似度，找到上一帧的实例与当前帧实例的映射关系，最终实现了实例的跟踪。

## introduction
* `tracking-by-detection`是目标解决多目标跟踪问题比较常用的方法。首先每一帧的物体都会被单独检测，一般检测结果为bounding box，然后再将两帧之间的实例进行关联。一般可以计算两个物体之间的相似度，来预测下一帧时刻物体的位置，但是这种方法通常需要知道运动模型，比如卡尔曼滤波或者视觉线索等。这在相机和物体同时移动的场景中难以应用，基于光流的`tracking-by-detection`模型不需要定义运动学模型，但是需要解决物体之间遮挡的问题。
* 作者之前提出了实例分割模型：(在论文Instance-aware Semantic Segmentation via Multi-task Network Cascades中)，用于进行实例分割，这个结果与基于bounding box的不同，分割结果不包含背景结构或者是其他物体的信息，因此可以提升相邻帧之间的相似度值计算的可靠性。

## contribution
* 作者首次实现基于实例分割结果的多目标跟踪，属于像素级别的，因而精细度更高。
* 论文中使用多种实例分割与多目标跟踪算法相结合，进行实验，对比结果。
* 之前光流法在应用的过程中，主要受到了bouding box中背景像素以及其他物体像素的影响，因而效果有时不好，本文则避免了这个问题。
* 论文中的模型即使对于高速相对运动的跟踪物体也有很好的跟踪效果。

## methods
* 在当前帧标注所有的实例及其index，在下一帧也计算这些index，计算两帧所有index之间的相似度，得到相似度矩阵，然后将映射成功的那一对实例赋予相同的trackid，相当于它们是同一个实例。对于匹配失败的实例，可以设置保存的帧数k，当连续k帧都没有匹配到这个物体，则将其从记录的集合中删除。
* 使用汉明距离计算相似度矩阵。

## evaluation
* 作者在`MOT 2D 2015 Benchmark Test Set`上进行实验，FasterRNN+SORT的MOTA最高，使用MNC的实例分割方法时，结合CPM的MOTA要高于SORT；在`MOT 2015 Benchmark KITTI-13`上实验时，使用了多种光流方法与MNC结合，设置不同的k值(上面提到的连续跟丢即抛弃实例的帧数)，k=1比k=0要获得更高的MOTA。CPM和DeepMatch的光流方法效果比PolyExp要好，因此光流法的选择也是很重要的。