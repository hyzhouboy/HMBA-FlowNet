from cv2 import reduce
import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from warp import tf_warp
import random

""" 
description
    utils tools
 """

def lrelu(x, leak=0.2, name='leaky_relu'):
    return tf.maximum(x, leak*x)


# patch: 32 x 32, 均匀划分卷积边缘图8*8个32*32的patch，选择熵值在一定阈值以内（统计个数）的随机5个样本位置并对应crop出warp error的patch
def edge_sampling_patch(x, img, sampling_num=5):
    p_w, p_h = 32
    threshold = 0.5
    filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                    [-2.0,-2.0,-2.0], [0,0,0], [2.0,2.0,2.0],
                                    [-1.0,-1.0,-1.0], [0,0,0], [1.0,1.0,1.0]],
                                    shape=[3,3,3,1]))    
    conv1 = tf.nn.conv2d(img, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
    norm = tf.cast(((conv1-tf.reduce_min(conv1))/(tf.reduce_max(conv1)-tf.reduce_min(conv1))), tf.float32)

    patch_warp_errors = []
    total_patch_warp_errors = []
    for idx in range(tf.shape(x)[0]):
        for i in range(tf.shape(x[idx])[1]/p_w):
            for j in range(tf.shape(x[idx])[0]/p_h):
                crop_x_ = tf.image.crop_to_bounding_box(x[idx], j*32, i*32, p_h, p_w)
                crop_edge_ = tf.image.crop_to_bounding_box(norm[idx], j*32, i*32, p_h, p_w)
                # patch_x = tf.reshape(crop_x_, [-1, H, W, d, d, channel])
                # patch_edge = tf.reshape(crop_edge_, [-1, H, W, 1, 1, channel])
                dot_ = tf.multiply(crop_x_, crop_edge_)
                cost_volume = tf.reduce_sum(dot_)
                if cost_volume >= threshold:
                    patch_warp_errors.append(crop_x_)


        total_patch_warp_errors.append(random.sample(patch_warp_errors, sampling_num))
            
    
    # 排序
    
    return total_patch_warp_errors

# patch: 32 x 32, 均匀划分退化warp error图里8*8个32*32的patch，选择清晰域边缘纹理筛选的5个样本区域以外的熵值前5个样本
def ranking_sampling_patch():
    img_size = tf.shape(x)
    p_w, p_h = 32
    filter = tf.Variable(tf.constant([[-1.0,-1.0,-1.0],[0,0,0],[1.0,1.0,1.0],
                                    [-2.0,-2.0,-2.0], [0,0,0], [2.0,2.0,2.0],
                                    [-1.0,-1.0,-1.0], [0,0,0], [1.0,1.0,1.0]],
                                    shape=[3,3,3,1]))    
    conv1 = tf.nn.conv2d(x, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
    norm = tf.cast(((conv1-tf.reduce_min(conv1))/(tf.reduce_max(conv1)-tf.reduce_min(conv1))), tf.float32)

    patch_warp_errors = []
    patch_warp_error_entropy = {}
    for i in range(30):
        rand_offset_h = tf.random_uniform([], 0, img_size[0]-p_h+1, dtype=tf.int32)
        rand_offset_w = tf.random_uniform([], 0, img_size[1]-p_w+1, dtype=tf.int32)
        crop_x_ = tf.image.crop_to_bounding_box(norm, rand_offset_h, rand_offset_w, p_h, p_w)
        patch_warp_errors.append(crop_x_)
        # 计算熵
        patch_warp_error_entropy[str(i)] = tf.reduce_mean(-tf.reduce_sum(crop_x_ * tf.log(crop_x_), axis=1))
    
    # 排序
    # sorted()
    return 0