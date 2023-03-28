
# -*- coding: utf-8 -*-
from cProfile import label
from turtle import shape

from matplotlib.pyplot import axis
import tensorflow as tf
import numpy as np
from helper import *

def epe_loss(diff, mask):
    diff_norm = tf.norm(diff, axis=-1, keepdims=True)
    diff_norm = tf.multiply(diff_norm, mask)
    diff_norm_sum = tf.reduce_sum(diff_norm)
    loss_mean = diff_norm_sum / (tf.reduce_sum(mask) + 1e-6)
        
    return loss_mean 

def abs_robust_loss(diff, mask, q=0.4):
    diff = tf.pow((tf.abs(diff)+0.01), q)
    diff = tf.multiply(diff, mask)
    diff_sum = tf.reduce_sum(diff)
    loss_mean = diff_sum / (tf.reduce_sum(mask) * 2 + 1e-6) 
    return loss_mean 
    
def create_mask(tensor, paddings):
    with tf.variable_scope('create_mask'):
        shape = tf.shape(tensor)
        inner_width = shape[1] - (paddings[0][0] + paddings[0][1])
        inner_height = shape[2] - (paddings[1][0] + paddings[1][1])
        inner = tf.ones([inner_width, inner_height])
    
        mask2d = tf.pad(inner, paddings)
        mask3d = tf.tile(tf.expand_dims(mask2d, 0), [shape[0], 1, 1])
        mask4d = tf.expand_dims(mask3d, 3)
        return tf.stop_gradient(mask4d) 
        
def census_loss(img1, img2_warped, mask, max_distance=3):
    patch_size = 2 * max_distance + 1
    with tf.variable_scope('census_loss'):
        def _ternary_transform(image):
            intensities = tf.image.rgb_to_grayscale(image) * 255
            #patches = tf.extract_image_patches( # fix rows_in is None
            #    intensities,
            #    ksizes=[1, patch_size, patch_size, 1],
            #    strides=[1, 1, 1, 1],
            #    rates=[1, 1, 1, 1],
            #    padding='SAME')
            out_channels = patch_size * patch_size
            w = np.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels))
            weights =  tf.constant(w, dtype=tf.float32)
            patches = tf.nn.conv2d(intensities, weights, strides=[1, 1, 1, 1], padding='SAME')
    
            transf = patches - intensities
            transf_norm = transf / tf.sqrt(0.81 + tf.square(transf))
            return transf_norm
    
        def _hamming_distance(t1, t2):
            dist = tf.square(t1 - t2)
            dist_norm = dist / (0.1 + dist)
            dist_sum = tf.reduce_sum(dist_norm, 3, keepdims=True)
            return dist_sum
    
        t1 = _ternary_transform(img1)
        t2 = _ternary_transform(img2_warped)
        dist = _hamming_distance(t1, t2)
    
        transform_mask = create_mask(mask, [[max_distance, max_distance],
                                                [max_distance, max_distance]])
        return abs_robust_loss(dist, mask * transform_mask) 

REAL_LABEL = 0.9
def discriminator_loss(D, y, fake_y, use_lsgan=True):
    """ Note: default: D(y).shape == (batch_size,5,5,1),
                       fake_buffer_size=50, batch_size=1
    Args:
      G: generator object
      D: discriminator object
      y: 4D tensor (batch_size, image_size, image_size, 3)
    Returns:
      loss: scalar
    """
    if use_lsgan:
        # use mean squared error
        error_real = tf.reduce_mean(tf.squared_difference(D(y), REAL_LABEL))
        error_fake = tf.reduce_mean(tf.square(D(fake_y)))
    else:
        # use cross entropy
        error_real = -tf.reduce_mean(safe_log(D(y)))
        error_fake = -tf.reduce_mean(safe_log(1-D(fake_y)))
    loss = (error_real + error_fake) / 2
    return loss


def generator_loss(D, fake_y, use_lsgan=True):
    """  fool discriminator into believing that G(x) is real
    """
    if use_lsgan:
        # use mean squared error
        loss = tf.reduce_mean(tf.squared_difference(D(fake_y), REAL_LABEL))
    else:
        # heuristic, non-saturating loss
        loss = -tf.reduce_mean(safe_log(D(fake_y))) / 2
    return loss


def cycle_consistency_loss(G, F, x, y, lambda1=10, lambda2=10):
    """ cycle consistency loss (L1 norm)
    """
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    forward_loss = tf.reduce_mean(tf.abs(F(G(x))-x))
    backward_loss = tf.reduce_mean(tf.abs(G(F(y))-y))
    loss = lambda1*forward_loss + lambda2*backward_loss
    return loss

def flow_consistency(flow_selfGT, flow_estimated, mask):
    diff = flow_selfGT - flow_estimated
    diff_norm = tf.norm(diff, axis=-1, keepdims=True)
    diff_norm = tf.multiply(diff_norm, mask)
    diff_norm_sum = tf.reduce_sum(diff_norm)
    loss_mean = diff_norm_sum / (tf.reduce_sum(mask) + 1e-6)
        
    return loss_mean 


def contra_NCE_loss(feat_e, feat_p, feat_n, nce_T=0.07):
    # 参考torch版本代码编写
    shape = tf.shape(feat_p)

    feat_e_ = tf.reshape(feat_e, [shape[0], 1, shape[1]])  # batchsize X 1 X 256
    feat_p_ = tf.reshape(feat_p, [shape[0], 1, shape[1]])  # batchsize X 1 X 256
    
    N = len(feat_n)
    feat_n_ = []
    for i in range(N):
        feat_n_.append(tf.reshape(feat_n[i], [shape[0], 1, shape[1]])) # batchsize X 1 X 256
    
    l_pos = tf.multiply(feat_e_, feat_p_)  # batchsize X 1 X 256

    # l_neg = tf.matmul(feat_e_, feat_n_, transpose_a=True)  # batchsize X 256 X 256
    l_neg = []
    for i in range(N):
        l_neg.append(tf.multiply(feat_e_, feat_n_[i]))  # batchsize X 3 X 256
   
    out = tf.concat([l_pos, l_neg[0]], axis=1) / nce_T  # batchsize X 2 X 256
    for i in range(N-1):
        out = tf.concat([out, l_neg[i+1]], axis=1) / nce_T  # batchsize X 4 X 256

    label = tf.zeros([shape[0], 1, shape[1]])  # batchsize X 1 X 256
    # label = tf.zeros([batchsize, 1, ])  # batchsize X 1 X 64
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out)  # optimize 每个采样都必须是0类正样本

    loss = tf.reduce_mean(loss)

    return loss

# 旧版本：5正样本和5负样本直接一起算
def patch_contra_NCE_loss(feat_e, feat_p, feat_n, nce_T=0.07):
    # 参考torch版本代码编写
    # N = len(feat_n)
    N = feat_n.get_shape().as_list()[0]
    # total_loss = 0
    shape = tf.shape(feat_p)

    # feat_e_ = []
    # feat_p_ = []
    feat_n_ = []
    # for i in range(N):
    #     feat_e_.append(tf.reshape(feat_e[i], [shape[0], 1, shape[1]]))  # batchsize X 1 X 256
    #     feat_p_.append(tf.reshape(feat_p[i], [shape[0], 1, shape[1]]))  # batchsize X 1 X 256
    #     feat_n_.append(tf.reshape(feat_n[i], [shape[0], 1, shape[1]]))  # batchsize X 1 X 256
    
    feat_e_ = tf.reshape(feat_e, [shape[0], 1, shape[1]])  # batchsize X 1 X 256
    feat_p_ = tf.reshape(feat_p, [shape[0], 1, shape[1]])  # batchsize X 1 X 256

    for i in range(N):
        feat_n_.append(tf.reshape(feat_n[i], [1, 1, shape[1]])) # batchsize X 1 X 1 X 256

    # feat_n_ = tf.reshape(feat_n, [shape[0], 1, shape[1]])  # batchsize X 1 X 256

    # for i in range(N):
    l_pos = tf.multiply(feat_e_, feat_p_)  # batchsize X 1 X 256

        # l_neg = tf.matmul(feat_e_, feat_n_, transpose_a=True)  # batchsize X 256 X 256
    # for i in range(N):
    l_neg = []
    for j in range(N):
        l_neg.append(tf.multiply(feat_p_, feat_n_[j]))  # batchsize X 5 X 256

    out = tf.concat([l_pos, l_neg[0]], axis=1)  # batchsize X 2 X 256
    for jj in range(N-1):
        out = tf.concat([out, l_neg[jj+1]], axis=1)  # batchsize X 6 X 256

    out /= nce_T

    label = tf.zeros([shape[0]*(N+1), 1, shape[1]])  # batchsize X 1 X 256
    # label = tf.zeros([batchsize, 1, ])  # batchsize X 1 X 64
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out)  # optimize 每个采样都必须是0类正样本

    loss = tf.reduce_mean(loss)

    # total_loss += loss
    
    # total_loss /= float(N)

    return loss


# 新版本：5正样本单独跟5负样本计算
def patch_contra_NCE_loss_one2multi(feat_e, feat_p, feat_n, nce_T=0.07):
    shape = tf.shape(feat_p)
    N = feat_p.get_shape().as_list()[0]
    feat_e_ = []
    feat_p_ = []
    feat_n_ = []

    for i in range(N):
        feat_e_.append(tf.reshape(feat_e[i], [1, 1, shape[1]]))
        feat_p_.append(tf.reshape(feat_p[i], [1, 1, shape[1]]))
        feat_n_.append(tf.reshape(feat_n[i], [1, 1, shape[1]]))
    
    label = tf.zeros([1, 1, shape[1]])
    average_loss = 0
    for i in range(N):
        l_pos = tf.multiply(feat_e_[i], feat_p_[i]) # 1 X 1 X 64
        out = tf.concat([l_pos, tf.multiply(feat_p_[i], feat_n_[0])], axis=1)
        for j in range(N-1):
            l_neg = tf.multiply(feat_p_[i], feat_n_[j+1])
            out = tf.concat([out, l_neg], axis=1)
        
        out /= nce_T

        loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=out)  # optimize 每个采样都必须是0类正样本
        loss = tf.reduce_mean(loss)
        average_loss += loss
    
    average_loss /= N 

    return average_loss


from warp import tf_warp
import layer
def feature_interaction(x_features, y_features, x_flow_estimated, y_flow_extimated, diff_cost, level=3):
    step = level
    d = 9
    loss = 0
    for i in range(step):
        # predict degradation cost volume
        feature_name = 'conv%d_2' % (i+1)
        x_shape = layer.get_shape(x_features['feature1'][feature_name], train=True)
        H = x_shape[1]
        W = x_shape[2]
        channel = x_shape[3]
        
        # compute x feature cost volume
        x_flow = x_flow_estimated['full_res']
        x_flow_ = layer.flow_resize(x_flow, tf.shape(x_features['feature1'][feature_name])[1:3], is_scale=True)
        x2_warp = tf_warp(x_features['feature2'][feature_name], x_flow_, H, W)
        # normalize
        x1 = tf.nn.l2_normalize(x_features['feature1'][feature_name], axis=3)
        x2_warp = tf.nn.l2_normalize(x2_warp, axis=3)  
        
        x2_patches = tf.extract_image_patches(x2_warp, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')

        x2_patches = tf.reshape(x2_patches, [-1, H, W, d, d, channel])
        x1_reshape = tf.reshape(x1, [-1, H, W, 1, 1, channel])
        x1_dot_x2 = tf.multiply(x1_reshape, x2_patches)
        x_cost_volume = tf.reduce_sum(x1_dot_x2, axis=-1)
        x_cost_volume = tf.reshape(x_cost_volume, [-1, H, W, d*d])

        # compute y feature cost volume
        y_flow = y_flow_extimated['full_res']
        y_flow_ = layer.flow_resize(y_flow, tf.shape(y_features['feature1'][feature_name])[1:3], is_scale=True)
        y2_warp = tf_warp(y_features['feature2'][feature_name], y_flow_, H, W)
        # normalize
        y1 = tf.nn.l2_normalize(y_features['feature1'][feature_name], axis=3)
        y2_warp = tf.nn.l2_normalize(y2_warp, axis=3)  
        d = 5
        y2_patches = tf.extract_image_patches(y2_warp, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')


        y2_patches = tf.reshape(y2_patches, [-1, H, W, d, d, channel])
        y1_reshape = tf.reshape(y1, [-1, H, W, 1, 1, channel])
        y1_dot_y2 = tf.multiply(y1_reshape, y2_patches)
        y_cost_volume = tf.reduce_sum(y1_dot_y2, axis=-1)
        y_cost_volume = tf.reshape(y_cost_volume, [-1, H, W, d*d])

        loss_ = y_cost_volume - x_cost_volume - diff_cost[feature_name]

        loss_ = tf.reduce_sum(loss_)
        loss += loss_
    
    loss /= step

    return loss