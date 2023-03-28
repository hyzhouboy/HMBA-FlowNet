# -*- coding: utf-8 -*-
from audioop import reverse
from turtle import shape
import tensorflow as tf
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from layer import get_shape
from datasets import BasicDataset, CycleFlow_Dataset
from utils import average_gradients, lrelu, occlusion, rgb_bgr, flow_warp_error, ImagePool
import utils
from warp import tf_warp

import random


# Degraded patch: 32 x 32, 均匀划分卷积边缘图8*8个32*32的patch，选择熵值在一定阈值以内（统计个数）的随机5个样本位置并对应crop出warp error的patch
# 退化域patch: 32 x 32, 均匀划分退化warp error图里8*8个32*32的patch，选择清晰域边缘纹理筛选的5个样本区域以外的熵值前5个样本
# strategy: random/ranking/edge_aware     mode: scene_x/scene_y
# scene_x: Clean images, scene_y：Degraded images
# 对于清晰策略：random/edge_aware， 对于退化图像：random/ranking
def scene_patch_sampling(img, pos1, pos2, neg, sampling_num=5, clean_strategy='random', degraded_strategy='random', mode='scene_x'):
    # img_size = tf.shape(img[0])
    img_size = get_shape(img[0])
    p_w = 32
    p_h = 32
    # threshold = 0.5
    threshold = tf.constant([0.5], shape=[1])
    with tf.variable_scope('sampling', reuse=False):
        filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                        [-2.0,-2.0,-2.0], [0,0,0], [2.0,2.0,2.0],
                                        [-1.0,-1.0,-1.0], [0,0,0], [1.0,1.0,1.0]],
                                        shape=[3,3,3,1]), trainable=False)    
        conv1 = tf.nn.conv2d(img, filter=filter, strides=[1, 1, 1, 1], padding='SAME')

    norm = tf.cast(((conv1-tf.reduce_min(conv1))/(tf.reduce_max(conv1)-tf.reduce_min(conv1))), tf.float32)

    pos1_patch_warp_errors = []
    pos2_patch_warp_errors = []
    neg_patch_warp_errors = []

    select_patch_warp_errors = {}

    if mode == 'scene_x':
        for idx in range(1):
            # 正样本采样
            if clean_strategy == 'edge_aware':
                for i in range(8):
                    for j in range(8):
                        crop_pos1_ = tf.image.crop_to_bounding_box(pos1[idx], j*32, i*32, p_h, p_w)
                        crop_edge_ = tf.image.crop_to_bounding_box(norm[idx], j*32, i*32, p_h, p_w)
                        crop_pos2_ = tf.image.crop_to_bounding_box(pos2[idx], j*32, i*32, p_h, p_w)
                        # patch_x = tf.reshape(crop_x_, [-1, H, W, d, d, channel])
                        # patch_edge = tf.reshape(crop_edge_, [-1, H, W, 1, 1, channel])
                        dot_ = tf.multiply(crop_pos1_, crop_edge_)
                        cost_volume = tf.reduce_sum(dot_)
                        # 边缘条件阈值可以去掉,*************************，加上是缩小选择范围
                        if tf.greater_equal(cost_volume, threshold):
                            pos1_patch_warp_errors.append(crop_pos1_)
                            pos2_patch_warp_errors.append(crop_pos2_)

                select_patch_warp_errors['pos_1'] = random.sample(pos1_patch_warp_errors, sampling_num)
                select_patch_warp_errors['pos_1'] = tf.reshape(select_patch_warp_errors['pos_1'], [sampling_num, p_h, p_w, 3])
                # select_patch_warp_errors['pos_1'] = tf.random_normal(select_patch_warp_errors['pos_1'])
                select_patch_warp_errors['pos_2'] = random.sample(pos2_patch_warp_errors, sampling_num)
                select_patch_warp_errors['pos_2'] = tf.reshape(select_patch_warp_errors['pos_2'], [sampling_num, p_h, p_w, 3])
            elif clean_strategy == 'random':
                for i in range(20):
                    rand_offset_h = tf.random_uniform([], 0, img_size[0]-p_h+1, dtype=tf.int32)
                    rand_offset_w = tf.random_uniform([], 0, img_size[1]-p_w+1, dtype=tf.int32)
                    crop_pos1_ = tf.image.crop_to_bounding_box(pos1[idx], rand_offset_h, rand_offset_w, p_h, p_w)
                    crop_pos2_ = tf.image.crop_to_bounding_box(pos2[idx], rand_offset_h, rand_offset_w, p_h, p_w)

                    pos1_patch_warp_errors.append(crop_pos1_)
                    pos2_patch_warp_errors.append(crop_pos2_)

                select_patch_warp_errors['pos_1'] = random.sample(pos1_patch_warp_errors, sampling_num)
                select_patch_warp_errors['pos_1'] = tf.reshape(select_patch_warp_errors['pos_1'], [sampling_num, p_h, p_w, 3])
                # select_patch_warp_errors['pos_1'] = tf.random_normal(select_patch_warp_errors['pos_1'])
                select_patch_warp_errors['pos_2'] = random.sample(pos2_patch_warp_errors, sampling_num)
                select_patch_warp_errors['pos_2'] = tf.reshape(select_patch_warp_errors['pos_2'], [sampling_num, p_h, p_w, 3])
            else:
                raise ValueError('Invalid strategy. Clean strategy should be one of {random, edge_aware} for clean scene')
                
        
            # 负样本采样
            # warp_error_entropy = {}
            warp_error_entropy = []
            for i in range(15):
                rand_offset_h = tf.random_uniform([], 0, img_size[0]-p_h+1, dtype=tf.int32)
                rand_offset_w = tf.random_uniform([], 0, img_size[1]-p_w+1, dtype=tf.int32)
                crop_neg_ = tf.image.crop_to_bounding_box(neg[idx], rand_offset_h, rand_offset_w, p_h, p_w)
                neg_patch_warp_errors.append(crop_neg_)
                # 计算熵
                warp_error_entropy.append(tf.reduce_mean(-tf.reduce_sum(crop_neg_ * tf.log(crop_neg_), axis=1)))
            
            if degraded_strategy == 'ranking':
                input_ = tf.reshape(warp_error_entropy, [15, 1])
                # 排序
                values_, indices = tf.nn.top(input_, 5)
                index = indices.as_list()
                select_neg = []
                for i__ in range(sampling_num):
                    select_neg.append(neg_patch_warp_errors[index[i__]])

                select_patch_warp_errors['neg'] = tf.reshape(select_neg, [sampling_num, p_h, p_w, 3])
                select_neg.clear()

                # select_patch_warp_errors['neg'] = random.sample(neg_patch_warp_errors, sampling_num)
                # select_patch_warp_errors['neg'] = tf.reshape(select_patch_warp_errors['neg'], [sampling_num, p_h, p_w, 3])
            elif degraded_strategy == 'random':
                select_patch_warp_errors['neg'] = random.sample(neg_patch_warp_errors, sampling_num)
                select_patch_warp_errors['neg'] = tf.reshape(select_patch_warp_errors['neg'], [sampling_num, p_h, p_w, 3])
            else:
                raise ValueError('Invalid strategy. Degraded strategy should be one of {random, ranking} for degraded scene')
        return select_patch_warp_errors
    
    elif mode == 'scene_y':
        for idx in range(1):
            # 正样本采样
            warp_error_entropy = []
            for i in range(15):
                rand_offset_h = tf.random_uniform([], 0, img_size[0]-p_h+1, dtype=tf.int32)
                rand_offset_w = tf.random_uniform([], 0, img_size[1]-p_w+1, dtype=tf.int32)
                crop_pos1_ = tf.image.crop_to_bounding_box(pos1[idx], rand_offset_h, rand_offset_w, p_h, p_w)
                crop_pos2_ = tf.image.crop_to_bounding_box(pos2[idx], rand_offset_h, rand_offset_w, p_h, p_w)
                pos1_patch_warp_errors.append(crop_pos1_)
                pos2_patch_warp_errors.append(crop_pos2_)
                # 计算熵
                # warp_error_entropy[i] = tf.reduce_mean(-tf.reduce_sum(crop_pos1_ * tf.log(crop_pos1_), axis=1))
                warp_error_entropy.append(tf.reduce_mean(-tf.reduce_sum(crop_pos1_ * tf.log(crop_pos1_), axis=1)))
            if degraded_strategy == 'ranking':
                input_ = tf.reshape(warp_error_entropy, [15, 1])
                # 排序
                values_, indices = tf.nn.top(input_, 5)
                index = indices.as_list()
                select_pos1 = []
                select_pos2 = []
                for i__ in range(sampling_num):
                    select_pos1.append(pos1_patch_warp_errors[index[i__]])
                    select_pos2.append(pos2_patch_warp_errors[index[i__]])

                select_patch_warp_errors['pos_1'] = tf.reshape(select_pos1, [sampling_num, p_h, p_w, 3])
                select_patch_warp_errors['pos_2'] = tf.reshape(select_pos2, [sampling_num, p_h, p_w, 3])
                select_pos1.clear()
                select_pos2.clear()

                # select_patch_warp_errors['pos_1'] = random.sample(pos1_patch_warp_errors, sampling_num)
                # select_patch_warp_errors['pos_2'] = random.sample(pos2_patch_warp_errors, sampling_num)
                # select_patch_warp_errors['pos_1'] = tf.reshape(select_patch_warp_errors['pos_1'], [sampling_num, p_h, p_w, 3])
                # select_patch_warp_errors['pos_2'] = tf.reshape(select_patch_warp_errors['pos_2'], [sampling_num, p_h, p_w, 3])
            elif degraded_strategy == 'random':
                select_patch_warp_errors['pos_1'] = random.sample(pos1_patch_warp_errors, sampling_num)
                select_patch_warp_errors['pos_2'] = random.sample(pos2_patch_warp_errors, sampling_num)
                select_patch_warp_errors['pos_1'] = tf.reshape(select_patch_warp_errors['pos_1'], [sampling_num, p_h, p_w, 3])
                select_patch_warp_errors['pos_2'] = tf.reshape(select_patch_warp_errors['pos_2'], [sampling_num, p_h, p_w, 3])
            else:
                raise ValueError('Invalid strategy. Degraded strategy should be one of {random, ranking} for degraded scene')

            # 负样本采样
            if clean_strategy == 'edge_aware':
                for i in range(8):
                    for j in range(8):
                        crop_neg_ = tf.image.crop_to_bounding_box(neg[idx], j*32, i*32, p_h, p_w)
                        crop_edge_ = tf.image.crop_to_bounding_box(norm[idx], j*32, i*32, p_h, p_w)
                        # patch_x = tf.reshape(crop_x_, [-1, H, W, d, d, channel])
                        # patch_edge = tf.reshape(crop_edge_, [-1, H, W, 1, 1, channel])
                        dot_ = tf.multiply(crop_neg_, crop_edge_)
                        cost_volume = tf.reduce_sum(dot_)
                        # if cost_volume >= threshold:
                        if tf.greater_equal(cost_volume, threshold):
                            neg_patch_warp_errors.append(crop_neg_)  
                
                select_patch_warp_errors['neg'] = random.sample(neg_patch_warp_errors, sampling_num)
                select_patch_warp_errors['neg'] = tf.reshape(select_patch_warp_errors['neg'], [sampling_num, p_h, p_w, 3])
            elif clean_strategy == 'random':
                for i in range(20):
                    rand_offset_h = tf.random_uniform([], 0, img_size[0]-p_h+1, dtype=tf.int32)
                    rand_offset_w = tf.random_uniform([], 0, img_size[1]-p_w+1, dtype=tf.int32)
                    crop_neg_ = tf.image.crop_to_bounding_box(neg[idx], rand_offset_h, rand_offset_w, p_h, p_w)

                    neg_patch_warp_errors.append(crop_neg_)

                select_patch_warp_errors['neg'] = random.sample(neg_patch_warp_errors, sampling_num)
                select_patch_warp_errors['neg'] = tf.reshape(select_patch_warp_errors['neg'], [sampling_num, p_h, p_w, 3])
            else:
                raise ValueError('Invalid strategy. Clean strategy should be one of {random, edge_aware} for clean scene')

        return select_patch_warp_errors 
    else:
        raise ValueError('Invalid mode. Sampling mode should be one of {scene_x, scene_y}.')
