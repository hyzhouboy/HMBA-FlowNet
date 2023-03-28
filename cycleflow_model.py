# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
import sys
import time
import cv2

from six.moves import xrange
from scipy import misc, io
from tensorflow.contrib import slim
from datetime import datetime

import matplotlib.pyplot as plt
from network import pyramid_processing, pyramid_processing_bidirection, Rain_pyramid_processing_flow, Generator, Discriminator, OpticalFlow
from layer import get_shape
from datasets import BasicDataset, CycleGan_BasicDataset
from utils import average_gradients, lrelu, occlusion, rgb_bgr, flow_warp_error, ImagePool
import utils
from data_augmentation import flow_resize
from flowlib import flow_to_color, write_flo
from warp import tf_warp
from loss import epe_loss, abs_robust_loss, census_loss, discriminator_loss, generator_loss, cycle_consistency_loss
import logging

from reader import Reader
""" 
description
    building model
    create data of one iterator, loss, train strategy, test stage
 """

class DDFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5, 
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False, 
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints', 
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_distillation", 
                 is_restore_model=False, restore_model='./models/KITTI/no_census_no_occlusion',
                 dataset_config={}, distillation_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1       
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.training_mode = training_mode
        self.is_restore_model = is_restore_model
        self.restore_model = restore_model
        self.dataset_config = dataset_config
        self.distillation_config = distillation_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))

        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)         
        
        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
        
        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))         
            
        self.sample_dir = '/'.join([self.save_dir, sample_dir])
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)  
        if not os.path.exists('/'.join([self.sample_dir, self.model_name])):
            os.makedirs(('/'.join([self.sample_dir, self.model_name])))    
        
        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir) 
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train']))) 
        if not os.path.exists('/'.join([self.summary_dir, 'test'])):
            os.makedirs(('/'.join([self.summary_dir, 'test'])))             
    
    def create_dataset_and_iterator(self, training_mode='no_distillation'):
        if training_mode=='no_distillation':
            dataset = BasicDataset(crop_h=self.dataset_config['crop_h'], 
                                   crop_w=self.dataset_config['crop_w'],
                                   batch_size=self.batch_size_per_gpu,
                                   data_list_file=self.dataset_config['data_list_file'],
                                   img_dir=self.dataset_config['img_dir'])
            iterator = dataset.create_batch_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)   
        elif training_mode == 'distillation':
            dataset = BasicDataset(crop_h=self.dataset_config['crop_h'], 
                                   crop_w=self.dataset_config['crop_w'],
                                   batch_size=self.batch_size_per_gpu,
                                   data_list_file=self.dataset_config['data_list_file'],
                                   img_dir=self.dataset_config['img_dir'],
				   fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
            iterator = dataset.create_batch_distillation_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads) 
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')
        return dataset, iterator
    
    
 
    def compute_losses(self, batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=True, is_scale=True):
        # un-supervised
        
        img_size = get_shape(batch_img1, train=train)
        img1_warp = tf_warp(batch_img1, flow_bw['full_res'], img_size[1], img_size[2])
        img2_warp = tf_warp(batch_img2, flow_fw['full_res'], img_size[1], img_size[2])
        
        losses = {}
        
        abs_robust_mean = {}
        abs_robust_mean['no_occlusion'] = abs_robust_loss(batch_img1-img2_warp, tf.ones_like(mask_fw)) + abs_robust_loss(batch_img2-img1_warp, tf.ones_like(mask_bw))
        abs_robust_mean['occlusion'] = abs_robust_loss(batch_img1-img2_warp, mask_fw) + abs_robust_loss(batch_img2-img1_warp, mask_bw)
        losses['abs_robust_mean'] = abs_robust_mean
        
        census_loss = {}
        census_loss['no_occlusion'] = census_loss(batch_img1, img2_warp, tf.ones_like(mask_fw), max_distance=3) + \
                    census_loss(batch_img2, img1_warp, tf.ones_like(mask_bw), max_distance=3)        
        census_loss['occlusion'] = census_loss(batch_img1, img2_warp, mask_fw, max_distance=3) + \
            census_loss(batch_img2, img1_warp, mask_bw, max_distance=3)
        losses['census'] = census_loss 
       


        # supervised
        # losses = {}
        # losses['no_occlusion'] = self.epe_loss(flow_fw['full_res']-flow_bw['full_res'], tf.ones_like(mask_fw))
        # epe_loss['occusion'] = self.epe_mask_loss(flow_fw-flow_bw, mask_fw)
        # losses['epe_loss'] = epe_loss_
        
        return losses
        
    def add_loss_summary(self, losses, keys=['abs_robust_mean'], prefix=None):
        for key in keys:
            for loss_key, loss_value in losses[key].items():
                if prefix:
                    loss_name = '%s/%s/%s' % (prefix, key, loss_key)
                else:
                    loss_name = '%s/%s' % (key, loss_key)
                tf.summary.scalar(loss_name, loss_value)
    
    def build_no_data_distillation(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True):
        # supervised
        """ 
        batch_img1_rgb, batch_img2_rgb, batch_img1_rain, batch_img2_rain, batch_img1_derain, batch_img2_derain, batch_flowgt = iterator.get_next()
        
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        
        flow_fw, flow_bw = Rain_pyramid_processing_flow(batch_img1_rgb, batch_img2_rgb, batch_img1_rain, batch_img2_rain, batch_img1_derain, batch_img2_derain, 
            train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)
        
        rgb_occ_fw, rgb_occ_bw = occlusion(flow_fw['rgb']['full_res'], flow_bw['rgb']['full_res'])
        rain_occ_fw, rain_occ_bw = occlusion(flow_fw['rain']['full_res'], flow_bw['rain']['full_res'])
        derain_occ_fw, derain_occ_bw = occlusion(flow_fw['derain']['full_res'], flow_bw['derain']['full_res'])

        rgb_mask_fw = 1. - rgb_occ_fw
        rgb_mask_bw = 1. - rgb_occ_bw  
        rain_mask_fw = 1. - rain_occ_fw
        rain_mask_bw = 1. - rain_occ_bw  
        derain_mask_fw = 1. - derain_occ_fw
        derain_mask_bw = 1. - derain_occ_bw  
        
        
        flow_gt = {}
        flow_gt['full_res'] = batch_flowgt

        losses = {}
        losses['rgb'] = self.compute_losses(batch_img1_rgb, batch_img2_rgb, flow_fw['rgb'], flow_gt, rgb_mask_fw, rgb_mask_bw, train=train, is_scale=is_scale)
        losses['rain'] = self.compute_losses(batch_img1_rain, batch_img2_rain, flow_fw['rain'], flow_gt, rain_mask_fw, rain_mask_bw, train=train, is_scale=is_scale)
        losses['derain'] = self.compute_losses(batch_img1_derain, batch_img2_derain, flow_fw['derain'], flow_gt, derain_mask_fw, derain_mask_bw, train=train, is_scale=is_scale)
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer)  
        """ 

        # un-supervised
        
        batch_img1, batch_img2 = iterator.get_next()
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        flow_fw, flow_bw = pyramid_processing_bidirection(batch_img1, batch_img2, 
            train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)  
        
        occ_fw, occ_bw = occlusion(flow_fw['full_res'], flow_bw['full_res'])
        mask_fw = 1. - occ_fw
        mask_bw = 1. - occ_bw  

        losses = self.compute_losses(batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=train, is_scale=is_scale)
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer) 
       


        
        return losses, regularizer_loss 
    
    def build_data_distillation(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True):
        batch_img1, batch_img2, flow_fw, flow_bw, occ_fw, occ_bw = iterator.get_next()
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        h = self.dataset_config['crop_h']
        w = self.dataset_config['crop_w']
        target_h = self.distillation_config['target_h']
        target_w = self.distillation_config['target_w']           
        offect_h = tf.random_uniform([], minval=0, maxval=h-target_h, dtype=tf.int32)
        offect_w = tf.random_uniform([], minval=0, maxval=w-target_w, dtype=tf.int32)
 
        
        batch_img1_cropped_patch = tf.image.crop_to_bounding_box(batch_img1, offect_h, offect_w, target_h, target_w)
        batch_img2_cropped_patch = tf.image.crop_to_bounding_box(batch_img2, offect_h, offect_w, target_h, target_w)     
        flow_fw_cropped_patch = tf.image.crop_to_bounding_box(flow_fw, offect_h, offect_w, target_h, target_w) 
        flow_bw_cropped_patch = tf.image.crop_to_bounding_box(flow_bw, offect_h, offect_w, target_h, target_w) 
        occ_fw_cropped_patch = tf.image.crop_to_bounding_box(occ_fw, offect_h, offect_w, target_h, target_w)
        occ_bw_cropped_patch = tf.image.crop_to_bounding_box(occ_bw, offect_h, offect_w, target_h, target_w)
        
        flow_fw_patch, flow_bw_patch = pyramid_processing_bidirection(batch_img1_cropped_patch, batch_img2_cropped_patch, 
            train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)  
        
        occ_fw_patch, occ_bw_patch = occlusion(flow_fw_patch['full_res'], flow_bw_patch['full_res'])
        mask_fw_patch = 1. - occ_fw_patch
        mask_bw_patch = 1. - occ_bw_patch

        losses = self.compute_losses(batch_img1_cropped_patch, batch_img2_cropped_patch, flow_fw_patch, flow_bw_patch, mask_fw_patch, mask_bw_patch, train=train, is_scale=is_scale)
        
        valid_mask_fw = tf.clip_by_value(occ_fw_patch - occ_fw_cropped_patch, 0., 1.)
        valid_mask_bw = tf.clip_by_value(occ_bw_patch - occ_bw_cropped_patch, 0., 1.)
        data_distillation_loss = {}
        data_distillation_loss['distillation'] = (abs_robust_loss(flow_fw_cropped_patch-flow_fw_patch['full_res'], valid_mask_fw) + \
                                       abs_robust_loss(flow_bw_cropped_patch-flow_bw_patch['full_res'], valid_mask_bw)) / 2
        losses['data_distillation'] = data_distillation_loss
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer)
        return losses, regularizer_loss  
    
    def build(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True, training_mode='no_distillation'):
        if training_mode == 'no_distillation':
            losses, regularizer_loss = self.build_no_data_distillation(iterator=iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale)
        elif training_mode == 'distillation':
            losses, regularizer_loss = self.build_data_distillation(iterator=iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale)
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')      
        return losses, regularizer_loss
                    
    def create_train_op(self, optim, iterator, global_step, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True, training_mode='no_distillation'):  
        if self.num_gpus == 1:
            losses, regularizer_loss = self.build(iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale, training_mode=training_mode)
            # supervised
            # optim_loss = losses['rgb']['no_occlusion'] + losses['rain']['no_occlusion'] + losses['derain']['no_occlusion']
            
            # optim_loss = losses['abs_robust_mean']['occlusion']
            optim_loss = losses['census']['occlusion'] + losses['data_distillation']['distillation']
            train_op = optim.minimize(optim_loss, var_list=tf.trainable_variables(), global_step=global_step)            
        else:
            tower_grads = []
            tower_losses = []
            tower_regularizer_losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_{}'.format(i)) as scope:
                            losses_, regularizer_loss_ = self.build(iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale, training_mode=training_mode) 
                            optim_loss = losses_['abs_robust_mean']['no_occlusion']

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            grads = self.optim.compute_gradients(optim_loss, var_list=tf.trainable_variables())
                            tower_grads.append(grads)
                            tower_losses.append(losses_)
                            tower_regularizer_losses.append(regularizer_loss_)
                            #self.add_loss_summary(losses_, keys=['abs_robust_mean', 'census'], prefix='tower_%d' % i)
                                        
            grads = average_gradients(tower_grads)
            train_op = optim.apply_gradients(grads, global_step=global_step)
            
            losses = tower_losses[0].copy()
            for key in losses.keys():
                for loss_key, loss_value in losses[key].items():
                    for i in range(1, self.num_gpus):
                        losses[key][loss_key] += tower_losses[i][key][loss_key]
                    losses[key][loss_key] /= self.num_gpus
            regularizer_loss = 0.
            for i in range(self.num_gpus):
                regularizer_loss += tower_regularizer_losses[i]
            regularizer_loss /= self.num_gpus

        self.add_loss_summary(losses, keys=losses.keys())
        tf.summary.scalar('regularizer_loss', regularizer_loss)
        
        return train_op, losses, regularizer_loss
    
    def train(self):
        with tf.Graph().as_default(), tf.device(self.shared_device):
            
            self.global_step = tf.Variable(0, trainable=False)
            self.dataset, self.iterator = self.create_dataset_and_iterator(training_mode=self.training_mode)       
            self.lr_decay = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', self.lr_decay)
            self.optim = tf.train.AdamOptimizer(self.lr_decay, self.beta1)            
            self.train_op, self.losses, self.regularizer_loss = self.create_train_op(optim=self.optim, iterator=self.iterator, 
                global_step=self.global_step, regularizer_scale=self.regularizer_scale, train=True, trainable=True, is_scale=self.is_scale, training_mode=self.training_mode)
            
            merge_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir='/'.join([self.summary_dir, 'train', self.model_name]))
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
            self.saver = tf.train.Saver(var_list=self.trainable_vars + [self.global_step], max_to_keep=500)
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement))
            
            self.sess.run(tf.global_variables_initializer())
            
            self.sess.run(tf.local_variables_initializer())
           
            if self.is_restore_model:
                self.saver.restore(self.sess, self.restore_model)
            
            self.sess.run(tf.assign(self.global_step, 0))
            start_step = self.sess.run(self.global_step)
            self.sess.run(self.iterator.initializer)
            start_time = time.time()
            for step in range(start_step+1, self.iter_steps+1):
                # supervised
                """ 
                _, rgb_epe, rain_epe, derain_epe = self.sess.run([self.train_op,
                    self.losses['rgb']['no_occlusion'], self.losses['rain']['no_occlusion'], self.losses['derain']['no_occlusion']])

                if np.mod(step, self.display_log_interval) == 0:
                    print('step: %d time: %.6fs, rgb_epe: %.6f, rain_epe: %.6f, derain_epe: %.6f' % 
                        (step, time.time() - start_time, rgb_epe, rain_epe, derain_epe)) 
                """
                # unsupervised
                
                _, abs_robust_mean_no_occlusion, census_occlusion = self.sess.run([self.train_op,
                    self.losses['abs_robust_mean']['no_occlusion'], self.losses['census']['occlusion']])
                if np.mod(step, self.display_log_interval) == 0:
                    print('step: %d time: %.6fs, abs_robust_mean_no_occlusion: %.6f, census_occlusion: %.6f' % 
                        (step, time.time() - start_time, abs_robust_mean_no_occlusion, census_occlusion)) 
               
                
                if np.mod(step, self.write_summary_interval) == 0:
                    summary_str = self.sess.run(merge_summary)
                    summary_writer.add_summary(summary_str, global_step=step)
                
                if np.mod(step, self.save_checkpoint_interval) == 0:
                    self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                    write_meta_graph=False, write_state=False) 
                    
    def test(self, restore_model, save_dir):
        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'], img_dir=self.dataset_config['img_dir'])
        save_name_list = dataset.data_list[:, 2]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img1, batch_img2 = iterator.get_next()
        flow_est = pyramid_processing(batch_img1, batch_img2, train=False, trainable=False, regularizer=None, is_scale=True) 

        # add error
        warp_error = flow_warp_error(batch_img1, batch_img2, flow_est['full_res']) 
        
        flow_est_color = flow_to_color(flow_est['full_res'], mask=None, max_flow=256)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(dataset.data_num):
            np_flow_est, np_flow_est_color, np_warp_error = sess.run([flow_est['full_res'], flow_est_color, warp_error])
            misc.imsave('%s/flow_est_color_%s.png' % (save_dir, save_name_list[i]), np_flow_est_color[0])
            # misc.imsave('%s/flow_warp_error_%s.png' % (save_dir, save_name_list[i]), np_warp_error[0])
            # write_flo('%s/flow_est_%s.flo' % (save_dir, save_name_list[i]), np_flow_est[0])
            print('Finish %d/%d' % (i, dataset.data_num))    
    
    def generate_fake_flow_occlusion(self, restore_model, save_dir):
        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'], img_dir=self.dataset_config['img_dir'])
        save_name_list = dataset.data_list[:, 2]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img1, batch_img2 = iterator.get_next()        
        flow_fw, flow_bw = pyramid_processing_bidirection(batch_img1, batch_img2, 
            train=False, trainable=False, reuse=None, regularizer=None, is_scale=True)  
        occ_fw, occ_bw = occlusion(flow_fw['full_res'], flow_bw['full_res'])        
        
        flow_fw_full_res = flow_fw['full_res'] * 64. + 32768
        flow_occ_fw = tf.concat([flow_fw_full_res, occ_fw], -1)
        flow_occ_fw = tf.cast(flow_occ_fw, tf.uint16)
        flow_bw_full_res = flow_bw['full_res'] * 64. + 32768
        flow_occ_bw = tf.concat([flow_bw_full_res, occ_bw], -1)
        flow_occ_bw = tf.cast(flow_occ_bw, tf.uint16)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        #save_dir = '/'.join([self.save_dir, 'sample', self.model_name])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        for i in range(dataset.data_num):
            np_flow_occ_fw, np_flow_occ_bw, np_occ_fw = sess.run([flow_occ_fw, flow_occ_bw, occ_fw])
            
            # opencv read and save image as bgr format, here we change rgb to bgr
            np_flow_occ_fw = rgb_bgr(np_flow_occ_fw[0])
            np_flow_occ_bw = rgb_bgr(np_flow_occ_bw[0])
            np_flow_occ_fw = np_flow_occ_fw.astype(np.uint16)
            np_flow_occ_bw = np_flow_occ_bw.astype(np.uint16)
            
            cv2.imwrite('%s/flow_occ_fw_%s.png' % (save_dir, save_name_list[i]), np_flow_occ_fw)
            cv2.imwrite('%s/flow_occ_bw_%s.png' % (save_dir, save_name_list[i]), np_flow_occ_bw)
            print('Finish %d/%d' % (i, dataset.data_num))            
            


# 改变
class CycleFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5, 
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False, 
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints', 
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_distillation", 
                 is_restore_model=False, restore_model='./models/KITTI/no_census_no_occlusion',
                 dataset_config={}, distillation_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1       
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.training_mode = training_mode
        self.is_restore_model = is_restore_model
        self.restore_model = restore_model
        self.dataset_config = dataset_config
        self.distillation_config = distillation_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))

        
        # 改动为cyclegan训练模式
        regularizer_ = slim.l2_regularizer(scale=self.regularizer_scale)
        self.is_train = True
        self.trainable = True
        self.reuse = None
        self.regularizer = regularizer_
        self.is_scale = True
        self.is_bidirection = True
        self.name = 'flow'
        self.FlowNet = OpticalFlow(train=self.is_train, trainable=self.trainable, reuse=self.reuse, regularizer=self.regularizer,
                                    is_scale=self.is_scale, is_bidirection=self.is_bidirection, name=self.name)


        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)         
        
        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
        
        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))         
            
        self.sample_dir = '/'.join([self.save_dir, sample_dir])
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)  
        if not os.path.exists('/'.join([self.sample_dir, self.model_name])):
            os.makedirs(('/'.join([self.sample_dir, self.model_name])))    
        
        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir) 
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train']))) 
        if not os.path.exists('/'.join([self.summary_dir, 'test'])):
            os.makedirs(('/'.join([self.summary_dir, 'test'])))             
    
    def create_dataset_and_iterator(self, training_mode='no_distillation'):
        if training_mode=='no_distillation':
            dataset = BasicDataset(crop_h=self.dataset_config['crop_h'], 
                                   crop_w=self.dataset_config['crop_w'],
                                   batch_size=self.batch_size_per_gpu,
                                   data_list_file=self.dataset_config['data_list_file'],
                                   img_dir=self.dataset_config['img_dir'])
            iterator = dataset.create_batch_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)   
        elif training_mode == 'distillation':
            dataset = BasicDataset(crop_h=self.dataset_config['crop_h'], 
                                   crop_w=self.dataset_config['crop_w'],
                                   batch_size=self.batch_size_per_gpu,
                                   data_list_file=self.dataset_config['data_list_file'],
                                   img_dir=self.dataset_config['img_dir'],
				   fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
            iterator = dataset.create_batch_distillation_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads) 
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')
        return dataset, iterator
    
    
 
    def compute_losses(self, batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=True, is_scale=True):
        # un-supervised
        
        img_size = get_shape(batch_img1, train=train)
        img1_warp = tf_warp(batch_img1, flow_bw['full_res'], img_size[1], img_size[2])
        img2_warp = tf_warp(batch_img2, flow_fw['full_res'], img_size[1], img_size[2])
        
        losses = {}
        
        abs_robust_mean = {}
        abs_robust_mean['no_occlusion'] = abs_robust_loss(batch_img1-img2_warp, tf.ones_like(mask_fw)) + abs_robust_loss(batch_img2-img1_warp, tf.ones_like(mask_bw))
        abs_robust_mean['occlusion'] = abs_robust_loss(batch_img1-img2_warp, mask_fw) + abs_robust_loss(batch_img2-img1_warp, mask_bw)
        losses['abs_robust_mean'] = abs_robust_mean
        
        census_loss_ = {}
        census_loss_['no_occlusion'] = census_loss(batch_img1, img2_warp, tf.ones_like(mask_fw), max_distance=3) + \
                    census_loss(batch_img2, img1_warp, tf.ones_like(mask_bw), max_distance=3)        
        census_loss_['occlusion'] = census_loss(batch_img1, img2_warp, mask_fw, max_distance=3) + \
            census_loss(batch_img2, img1_warp, mask_bw, max_distance=3)
        losses['census'] = census_loss_ 
       


        # supervised
        # losses = {}
        # losses['no_occlusion'] = self.epe_loss(flow_fw['full_res']-flow_bw['full_res'], tf.ones_like(mask_fw))
        # epe_loss['occusion'] = self.epe_mask_loss(flow_fw-flow_bw, mask_fw)
        # losses['epe_loss'] = epe_loss_
        
        return losses
        
    def add_loss_summary(self, losses, keys=['abs_robust_mean'], prefix=None):
        for key in keys:
            for loss_key, loss_value in losses[key].items():
                if prefix:
                    loss_name = '%s/%s/%s' % (prefix, key, loss_key)
                else:
                    loss_name = '%s/%s' % (key, loss_key)
                tf.summary.scalar(loss_name, loss_value)
    
    def build_no_data_distillation(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True):
        # supervised
        """ 
        batch_img1_rgb, batch_img2_rgb, batch_img1_rain, batch_img2_rain, batch_img1_derain, batch_img2_derain, batch_flowgt = iterator.get_next()
        
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        
        flow_fw, flow_bw = Rain_pyramid_processing_flow(batch_img1_rgb, batch_img2_rgb, batch_img1_rain, batch_img2_rain, batch_img1_derain, batch_img2_derain, 
            train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)
        
        rgb_occ_fw, rgb_occ_bw = occlusion(flow_fw['rgb']['full_res'], flow_bw['rgb']['full_res'])
        rain_occ_fw, rain_occ_bw = occlusion(flow_fw['rain']['full_res'], flow_bw['rain']['full_res'])
        derain_occ_fw, derain_occ_bw = occlusion(flow_fw['derain']['full_res'], flow_bw['derain']['full_res'])

        rgb_mask_fw = 1. - rgb_occ_fw
        rgb_mask_bw = 1. - rgb_occ_bw  
        rain_mask_fw = 1. - rain_occ_fw
        rain_mask_bw = 1. - rain_occ_bw  
        derain_mask_fw = 1. - derain_occ_fw
        derain_mask_bw = 1. - derain_occ_bw  
        
        
        flow_gt = {}
        flow_gt['full_res'] = batch_flowgt

        losses = {}
        losses['rgb'] = self.compute_losses(batch_img1_rgb, batch_img2_rgb, flow_fw['rgb'], flow_gt, rgb_mask_fw, rgb_mask_bw, train=train, is_scale=is_scale)
        losses['rain'] = self.compute_losses(batch_img1_rain, batch_img2_rain, flow_fw['rain'], flow_gt, rain_mask_fw, rain_mask_bw, train=train, is_scale=is_scale)
        losses['derain'] = self.compute_losses(batch_img1_derain, batch_img2_derain, flow_fw['derain'], flow_gt, derain_mask_fw, derain_mask_bw, train=train, is_scale=is_scale)
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer)  
        """ 

        # un-supervised
        
        batch_img1, batch_img2 = iterator.get_next()
        
        flow_fw, flow_bw = self.FlowNet([batch_img1, batch_img2])
        
        # regularizer = slim.l2_regularizer(scale=regularizer_scale)
        # flow_fw, flow_bw = pyramid_processing_bidirection(batch_img1, batch_img2, 
        #     train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)  
        
        occ_fw, occ_bw = occlusion(flow_fw['full_res'], flow_bw['full_res'])
        mask_fw = 1. - occ_fw
        mask_bw = 1. - occ_bw  

        losses = self.compute_losses(batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=train, is_scale=is_scale)
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer) 
       


        
        return losses, regularizer_loss 
    
    def build_data_distillation(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True):
        batch_img1, batch_img2, flow_fw, flow_bw, occ_fw, occ_bw = iterator.get_next()
        regularizer = slim.l2_regularizer(scale=regularizer_scale)
        h = self.dataset_config['crop_h']
        w = self.dataset_config['crop_w']
        target_h = self.distillation_config['target_h']
        target_w = self.distillation_config['target_w']           
        offect_h = tf.random_uniform([], minval=0, maxval=h-target_h, dtype=tf.int32)
        offect_w = tf.random_uniform([], minval=0, maxval=w-target_w, dtype=tf.int32)
 
        
        batch_img1_cropped_patch = tf.image.crop_to_bounding_box(batch_img1, offect_h, offect_w, target_h, target_w)
        batch_img2_cropped_patch = tf.image.crop_to_bounding_box(batch_img2, offect_h, offect_w, target_h, target_w)     
        flow_fw_cropped_patch = tf.image.crop_to_bounding_box(flow_fw, offect_h, offect_w, target_h, target_w) 
        flow_bw_cropped_patch = tf.image.crop_to_bounding_box(flow_bw, offect_h, offect_w, target_h, target_w) 
        occ_fw_cropped_patch = tf.image.crop_to_bounding_box(occ_fw, offect_h, offect_w, target_h, target_w)
        occ_bw_cropped_patch = tf.image.crop_to_bounding_box(occ_bw, offect_h, offect_w, target_h, target_w)
        
        flow_fw_patch, flow_bw_patch = self.FlowNet([batch_img1_cropped_patch, batch_img2_cropped_patch])
        
        # flow_fw_patch, flow_bw_patch = pyramid_processing_bidirection(batch_img1_cropped_patch, batch_img2_cropped_patch, 
        #     train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)  
        
        occ_fw_patch, occ_bw_patch = occlusion(flow_fw_patch['full_res'], flow_bw_patch['full_res'])
        mask_fw_patch = 1. - occ_fw_patch
        mask_bw_patch = 1. - occ_bw_patch

        losses = self.compute_losses(batch_img1_cropped_patch, batch_img2_cropped_patch, flow_fw_patch, flow_bw_patch, mask_fw_patch, mask_bw_patch, train=train, is_scale=is_scale)
        
        valid_mask_fw = tf.clip_by_value(occ_fw_patch - occ_fw_cropped_patch, 0., 1.)
        valid_mask_bw = tf.clip_by_value(occ_bw_patch - occ_bw_cropped_patch, 0., 1.)
        data_distillation_loss = {}
        data_distillation_loss['distillation'] = (abs_robust_loss(flow_fw_cropped_patch-flow_fw_patch['full_res'], valid_mask_fw) + \
                                       abs_robust_loss(flow_bw_cropped_patch-flow_bw_patch['full_res'], valid_mask_bw)) / 2
        losses['data_distillation'] = data_distillation_loss
        
        l2_regularizer = tf.losses.get_regularization_losses()
        regularizer_loss = tf.add_n(l2_regularizer)
        return losses, regularizer_loss  
    
    def build(self, iterator, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True, training_mode='no_distillation'):
        if training_mode == 'no_distillation':
            losses, regularizer_loss = self.build_no_data_distillation(iterator=iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale)
        elif training_mode == 'distillation':
            losses, regularizer_loss = self.build_data_distillation(iterator=iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale)
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')      
        return losses, regularizer_loss
                    
    def create_train_op(self, optim, iterator, global_step, regularizer_scale=1e-4, train=True, trainable=True, is_scale=True, training_mode='no_distillation'):  
        if self.num_gpus == 1:
            losses, regularizer_loss = self.build(iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale, training_mode=training_mode)
            # supervised
            # optim_loss = losses['rgb']['no_occlusion'] + losses['rain']['no_occlusion'] + losses['derain']['no_occlusion']
            
            # optim_loss = losses['abs_robust_mean']['occlusion']
            optim_loss = losses['census']['occlusion'] + losses['data_distillation']['distillation']
            train_op = optim.minimize(optim_loss, var_list=tf.trainable_variables(), global_step=global_step)            
        else:
            tower_grads = []
            tower_losses = []
            tower_regularizer_losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(self.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_{}'.format(i)) as scope:
                            losses_, regularizer_loss_ = self.build(iterator, regularizer_scale=regularizer_scale, train=train, trainable=trainable, is_scale=is_scale, training_mode=training_mode) 
                            optim_loss = losses_['abs_robust_mean']['no_occlusion']

                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()

                            grads = self.optim.compute_gradients(optim_loss, var_list=tf.trainable_variables())
                            tower_grads.append(grads)
                            tower_losses.append(losses_)
                            tower_regularizer_losses.append(regularizer_loss_)
                            #self.add_loss_summary(losses_, keys=['abs_robust_mean', 'census'], prefix='tower_%d' % i)
                                        
            grads = average_gradients(tower_grads)
            train_op = optim.apply_gradients(grads, global_step=global_step)
            
            losses = tower_losses[0].copy()
            for key in losses.keys():
                for loss_key, loss_value in losses[key].items():
                    for i in range(1, self.num_gpus):
                        losses[key][loss_key] += tower_losses[i][key][loss_key]
                    losses[key][loss_key] /= self.num_gpus
            regularizer_loss = 0.
            for i in range(self.num_gpus):
                regularizer_loss += tower_regularizer_losses[i]
            regularizer_loss /= self.num_gpus

        self.add_loss_summary(losses, keys=losses.keys())
        tf.summary.scalar('regularizer_loss', regularizer_loss)
        
        return train_op, losses, regularizer_loss
    
    def train(self):
        with tf.Graph().as_default(), tf.device(self.shared_device):
            
            self.global_step = tf.Variable(0, trainable=False)
            self.dataset, self.iterator = self.create_dataset_and_iterator(training_mode=self.training_mode)       
            self.lr_decay = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
            tf.summary.scalar('learning_rate', self.lr_decay)
            self.optim = tf.train.AdamOptimizer(self.lr_decay, self.beta1)            
            self.train_op, self.losses, self.regularizer_loss = self.create_train_op(optim=self.optim, iterator=self.iterator, 
                global_step=self.global_step, regularizer_scale=self.regularizer_scale, train=True, trainable=True, is_scale=self.is_scale, training_mode=self.training_mode)
            
            merge_summary = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(logdir='/'.join([self.summary_dir, 'train', self.model_name]))
            self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
            self.saver = tf.train.Saver(var_list=self.trainable_vars + [self.global_step], max_to_keep=500)
            
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement))
            
            self.sess.run(tf.global_variables_initializer())
            
            self.sess.run(tf.local_variables_initializer())
           
            if self.is_restore_model:
                self.saver.restore(self.sess, self.restore_model)
            
            self.sess.run(tf.assign(self.global_step, 0))
            start_step = self.sess.run(self.global_step)
            self.sess.run(self.iterator.initializer)
            start_time = time.time()
            for step in range(start_step+1, self.iter_steps+1):
                # supervised
                """ 
                _, rgb_epe, rain_epe, derain_epe = self.sess.run([self.train_op,
                    self.losses['rgb']['no_occlusion'], self.losses['rain']['no_occlusion'], self.losses['derain']['no_occlusion']])

                if np.mod(step, self.display_log_interval) == 0:
                    print('step: %d time: %.6fs, rgb_epe: %.6f, rain_epe: %.6f, derain_epe: %.6f' % 
                        (step, time.time() - start_time, rgb_epe, rain_epe, derain_epe)) 
                """
                # unsupervised
                
                _, abs_robust_mean_no_occlusion, census_occlusion = self.sess.run([self.train_op,
                    self.losses['abs_robust_mean']['no_occlusion'], self.losses['census']['occlusion']])
                if np.mod(step, self.display_log_interval) == 0:
                    print('step: %d time: %.6fs, abs_robust_mean_no_occlusion: %.6f, census_occlusion: %.6f' % 
                        (step, time.time() - start_time, abs_robust_mean_no_occlusion, census_occlusion)) 
               
                
                if np.mod(step, self.write_summary_interval) == 0:
                    summary_str = self.sess.run(merge_summary)
                    summary_writer.add_summary(summary_str, global_step=step)
                
                if np.mod(step, self.save_checkpoint_interval) == 0:
                    self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                    write_meta_graph=False, write_state=False) 
                    
    def test(self, restore_model, save_dir):

        flownet = OpticalFlow(train=False, trainable=False, reuse=True, regularizer=None, is_scale=True, is_bidirection=False, name='flow')

        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'], img_dir=self.dataset_config['img_dir'])
        save_name_list = dataset.data_list[:, 2]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img1, batch_img2 = iterator.get_next()

        # flow_est = pyramid_processing(batch_img1, batch_img2, train=False, trainable=False, regularizer=None, is_scale=True) 
        flow_est = flownet([batch_img1, batch_img2])

        # add error
        warp_error = flow_warp_error(batch_img1, batch_img2, flow_est['full_res']) 
        
        flow_est_color = flow_to_color(flow_est['full_res'], mask=None, max_flow=256)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        for i in range(dataset.data_num):
            np_flow_est, np_flow_est_color, np_warp_error = sess.run([flow_est['full_res'], flow_est_color, warp_error])
            misc.imsave('%s/flow_est_color_%s.png' % (save_dir, save_name_list[i]), np_flow_est_color[0])
            # misc.imsave('%s/flow_warp_error_%s.png' % (save_dir, save_name_list[i]), np_warp_error[0])
            # write_flo('%s/flow_est_%s.flo' % (save_dir, save_name_list[i]), np_flow_est[0])
            print('Finish %d/%d' % (i, dataset.data_num))    
    
    def generate_fake_flow_occlusion(self, restore_model, save_dir):

        flownet = OpticalFlow(train=False, trainable=False, reuse=None, regularizer=None, is_scale=True, is_bidirection=True, name='flow')

        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'], img_dir=self.dataset_config['img_dir'])
        save_name_list = dataset.data_list[:, 2]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img1, batch_img2 = iterator.get_next()   
        
             
        # flow_fw, flow_bw = pyramid_processing_bidirection(batch_img1, batch_img2, 
        #     train=False, trainable=False, reuse=None, regularizer=None, is_scale=True)  

        flow_fw, flow_bw = flownet([batch_img1, batch_img2])
        occ_fw, occ_bw = occlusion(flow_fw['full_res'], flow_bw['full_res'])        
        
        flow_fw_full_res = flow_fw['full_res'] * 64. + 32768
        flow_occ_fw = tf.concat([flow_fw_full_res, occ_fw], -1)
        flow_occ_fw = tf.cast(flow_occ_fw, tf.uint16)
        flow_bw_full_res = flow_bw['full_res'] * 64. + 32768
        flow_occ_bw = tf.concat([flow_bw_full_res, occ_bw], -1)
        flow_occ_bw = tf.cast(flow_occ_bw, tf.uint16)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        #save_dir = '/'.join([self.save_dir, 'sample', self.model_name])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        for i in range(dataset.data_num):
            np_flow_occ_fw, np_flow_occ_bw, np_occ_fw = sess.run([flow_occ_fw, flow_occ_bw, occ_fw])
            
            # opencv read and save image as bgr format, here we change rgb to bgr
            np_flow_occ_fw = rgb_bgr(np_flow_occ_fw[0])
            np_flow_occ_bw = rgb_bgr(np_flow_occ_bw[0])
            np_flow_occ_fw = np_flow_occ_fw.astype(np.uint16)
            np_flow_occ_bw = np_flow_occ_bw.astype(np.uint16)
            
            cv2.imwrite('%s/flow_occ_fw_%s.png' % (save_dir, save_name_list[i]), np_flow_occ_fw)
            cv2.imwrite('%s/flow_occ_bw_%s.png' % (save_dir, save_name_list[i]), np_flow_occ_bw)
            print('Finish %d/%d' % (i, dataset.data_num))            
            



# cyclegan整合测试, 先整合到跟光流结构一致

class CycleGAN:
    def __init__(self,
                X_train_file='',
                Y_train_file='',
                batch_size=1,
                image_size=256,
                use_lsgan=True,
                norm='instance',
                lambda1=10,
                lambda2=10,
                learning_rate=2e-4,
                beta1=0.5,
                ngf=64,
                load_model='',
                pool_size=50,
                buffer_size=1000,
                num_input_threads=4,
                x_data_list_file='',
                y_data_list_file='',
                img_dir=''
                ):
        """
        Args:
        X_train_file: string, X tfrecords file for training
        Y_train_file: string Y tfrecords file for training
        batch_size: integer, batch size
        image_size: integer, image size
        lambda1: integer, weight for forward cycle loss (X->Y->X)
        lambda2: integer, weight for backward cycle loss (Y->X->Y)
        use_lsgan: boolean
        norm: 'instance' or 'batch'
        learning_rate: float, initial learning rate for Adam
        beta1: float, momentum term of Adam
        ngf: number of gen filters in first conv layer
        """
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.X_train_file = X_train_file
        self.Y_train_file = Y_train_file
        self.load_model = load_model
        self.pool_size = pool_size

        self.buffer_size = buffer_size
        self.num_input_threads = num_input_threads
        self.x_data_list_file = x_data_list_file
        self.y_data_list_file = y_data_list_file
        self.img_dir = img_dir

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')
        # self.allow_soft_placement=allow_soft_placement 
        # self.log_device_placement=log_device_placement
        # 

        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_size=image_size)
        self.D_Y = Discriminator('D_Y',
            self.is_training, norm=norm, use_sigmoid=use_sigmoid)
        self.F = Generator('F', self.is_training, norm=norm, image_size=image_size)
        self.D_X = Discriminator('D_X',
            self.is_training, norm=norm, use_sigmoid=use_sigmoid)

        # self.fake_x = tf.placeholder(tf.float32,
        #     shape=[batch_size, image_size, image_size, 3])
        # self.fake_y = tf.placeholder(tf.float32,
        #     shape=[batch_size, image_size, image_size, 3])
        
        if self.load_model is not None:
            self.checkpoints_dir = "checkpoints/" + self.load_model.lstrip("checkpoints/")
        else:
            current_time = datetime.now().strftime("%Y%m%d-%H%M")
            self.checkpoints_dir = "checkpoints/{}".format(current_time)
            try:
                os.makedirs(self.checkpoints_dir)
            except os.error:
                pass

    # iterator
    def create_dataset_and_iterator(self):
        
        x_dataset = CycleGan_BasicDataset(crop_h=self.image_size, 
                                crop_w=self.image_size,
                                batch_size=self.batch_size,
                                data_list_file=self.x_data_list_file,
                                img_dir=self.img_dir)
                                
        x_iterator = x_dataset.create_batch_iterator(data_list=x_dataset.data_list, batch_size=x_dataset.batch_size,
            shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)  

        y_dataset = CycleGan_BasicDataset(crop_h=self.image_size, 
                                crop_w=self.image_size,
                                batch_size=self.batch_size,
                                data_list_file=self.y_data_list_file,
                                img_dir=self.img_dir)
        y_iterator = y_dataset.create_batch_iterator(data_list=y_dataset.data_list, batch_size=y_dataset.batch_size,
            shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)    
        
        return x_dataset, x_iterator, y_dataset, y_iterator

    def model(self):
        
        self.fake_x = tf.placeholder(tf.float32,
            shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.fake_y = tf.placeholder(tf.float32,
            shape=[self.batch_size, self.image_size, self.image_size, 3])

        X_reader = Reader(self.X_train_file, name='X',
            image_size=self.image_size, batch_size=self.batch_size)
        Y_reader = Reader(self.Y_train_file, name='Y',
            image_size=self.image_size, batch_size=self.batch_size)

        x = X_reader.feed()
        y = Y_reader.feed()

        cycle_loss = cycle_consistency_loss(self.G, self.F, x, y, self.lambda1, self.lambda2)

        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        G_loss =  G_gan_loss + cycle_loss
        D_Y_loss = discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)
        # D_Y_loss = discriminator_loss(self.D_Y, y, fake_y, use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(y)
        F_gan_loss = generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)
        # D_X_loss = discriminator_loss(self.D_X, x, fake_x, use_lsgan=self.use_lsgan)

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)

        tf.summary.image('X/generated', utils.batch_convert2int(self.G(x)))
        tf.summary.image('X/reconstruction', utils.batch_convert2int(self.F(self.G(x))))
        tf.summary.image('Y/generated', utils.batch_convert2int(self.F(y)))
        tf.summary.image('Y/reconstruction', utils.batch_convert2int(self.G(self.F(y))))

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x
    

    # 直接进行图像读取
    def model_no_tfrecorder(self, x_iterator, y_iterator):
        
        self.fake_x = tf.placeholder(tf.float32,
            shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.fake_y = tf.placeholder(tf.float32,
            shape=[self.batch_size, self.image_size, self.image_size, 3])


        # # 输入修改
        # X_reader = Reader(self.X_train_file, name='X',
        #     image_size=self.image_size, batch_size=self.batch_size)
        # Y_reader = Reader(self.Y_train_file, name='Y',
        #     image_size=self.image_size, batch_size=self.batch_size)

        # # 此处的输入是tf解码后的图像buffer, 修改
        # x = X_reader.feed()
        # y = Y_reader.feed()
        # [1, 256, 256, 3]

        # 修改
        
        x = x_iterator.get_next()
        y = y_iterator.get_next()

        tf.summary.image('X/X_input', x)   
        tf.summary.image('Y/Y_input', y)   
        
        
        
        cycle_loss = cycle_consistency_loss(self.G, self.F, x, y, self.lambda1, self.lambda2)

        # X -> Y
        fake_y = self.G(x)
        G_gan_loss = generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        G_loss =  G_gan_loss + cycle_loss
        D_Y_loss = discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)
        # D_Y_loss = discriminator_loss(self.D_Y, y, fake_y, use_lsgan=self.use_lsgan)

        # Y -> X
        fake_x = self.F(y)
        F_gan_loss = generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)
        # D_X_loss = discriminator_loss(self.D_X, x, fake_x, use_lsgan=self.use_lsgan)

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x)))
        tf.summary.histogram('D_X/true', self.D_X(x))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)

        tf.summary.image('X/generated', self.G(x))
        tf.summary.image('X/reconstruction', self.F(self.G(x)))
        tf.summary.image('Y/generated', self.F(y))
        tf.summary.image('Y/reconstruction', self.G(self.F(y)))

        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x

    def optimize(self, G_loss, D_Y_loss, F_loss, D_X_loss):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                        tf.greater_equal(global_step, start_decay_step),
                        tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                    decay_steps, end_learning_rate,
                                                    power=1.0),
                        starter_learning_rate
                )

            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1=beta1, name=name)
                        .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

        with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
            return tf.no_op(name='optimizers')
    
    def train(self):
        graph = tf.Graph()
        with graph.as_default():
            # cycle_gan = CycleGAN(
            #     X_train_file=FLAGS.X,
            #     Y_train_file=FLAGS.Y,
            #     batch_size=FLAGS.batch_size,
            #     image_size=FLAGS.image_size,
            #     use_lsgan=FLAGS.use_lsgan,
            #     norm=FLAGS.norm,
            #     lambda1=FLAGS.lambda1,
            #     lambda2=FLAGS.lambda2,
            #     learning_rate=FLAGS.learning_rate,
            #     beta1=FLAGS.beta1,
            #     ngf=FLAGS.ngf
            # )
            G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = self.model()
            optimizers = self.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.checkpoints_dir, graph)
            saver = tf.train.Saver()

        # with tf.Session(config=tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement)) as sess:
        with tf.Session(graph=graph) as sess:
            if self.load_model is not None:
                checkpoint = tf.train.get_checkpoint_state(self.checkpoints_dir)
                meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(self.checkpoints_dir))
                step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                step = 0

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                fake_Y_pool = ImagePool(self.pool_size)
                fake_X_pool = ImagePool(self.pool_size)

                while not coord.should_stop():
                    # get previously generated images
                    fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

                    # train
                    _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
                        sess.run(
                            [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                            feed_dict={self.fake_y: fake_Y_pool.query(fake_y_val),
                                        self.fake_x: fake_X_pool.query(fake_x_val)}
                        )
                    )

                    # train_writer.add_summary(summary, step)
                    # train_writer.flush()

                    if step % 100 == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info('  G_loss   : {}'.format(G_loss_val))
                        logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                        logging.info('  F_loss   : {}'.format(F_loss_val))
                        logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                        train_writer.add_summary(summary, step)
                        train_writer.flush()

                    if step % 10000 == 0:
                        save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                    step += 1

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)



    # no-tfrecord version
    def train_notf(self):
        graph = tf.Graph()
        with graph.as_default():
            # cycle_gan = CycleGAN(
            #     X_train_file=FLAGS.X,
            #     Y_train_file=FLAGS.Y,
            #     batch_size=FLAGS.batch_size,
            #     image_size=FLAGS.image_size,
            #     use_lsgan=FLAGS.use_lsgan,
            #     norm=FLAGS.norm,
            #     lambda1=FLAGS.lambda1,
            #     lambda2=FLAGS.lambda2,
            #     learning_rate=FLAGS.learning_rate,
            #     beta1=FLAGS.beta1,
            #     ngf=FLAGS.ngf
            # )
            self.global_step = tf.Variable(0, trainable=False)
            self.X_dataset, self.X_iterator, self.Y_dataset, self.Y_iterator = self.create_dataset_and_iterator()

            G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x = self.model_no_tfrecorder(self.X_iterator, self.Y_iterator)
            optimizers = self.optimize(G_loss, D_Y_loss, F_loss, D_X_loss)

            summary_op = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.checkpoints_dir, graph)
            saver = tf.train.Saver()

            

        # with tf.Session(config=tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement)) as sess:
        with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            if self.load_model is not None:
                checkpoint = tf.train.get_checkpoint_state(self.checkpoints_dir)
                meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                restore = tf.train.import_meta_graph(meta_graph_path)
                restore.restore(sess, tf.train.latest_checkpoint(self.checkpoints_dir))
                step = int(meta_graph_path.split("-")[2].split(".")[0])
            else:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                step = 0

            # sess.run(tf.global_variables_initializer())
            
            # step = 0

            sess.run(tf.assign(self.global_step, 0))
            start_step = sess.run(self.global_step)
            sess.run(self.X_iterator.initializer) 
            sess.run(self.Y_iterator.initializer) 
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                fake_Y_pool = ImagePool(self.pool_size)
                fake_X_pool = ImagePool(self.pool_size)

                
                while not coord.should_stop():
                    # get previously generated images
                    fake_y_val, fake_x_val = sess.run([fake_y, fake_x])

                    # train
                    _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, summary = (
                        sess.run(
                            [optimizers, G_loss, D_Y_loss, F_loss, D_X_loss, summary_op],
                            feed_dict={self.fake_y: fake_Y_pool.query(fake_y_val),
                                        self.fake_x: fake_X_pool.query(fake_x_val)}
                        )
                    )

                    # train_writer.add_summary(summary, step)
                    # train_writer.flush()

                    if step % 100 == 0:
                        logging.info('-----------Step %d:-------------' % step)
                        logging.info('  G_loss   : {}'.format(G_loss_val))
                        logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                        logging.info('  F_loss   : {}'.format(F_loss_val))
                        logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                        train_writer.add_summary(summary, step)
                        train_writer.flush()

                    if step % 10000 == 0:
                        save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                        logging.info("Model saved in file: %s" % save_path)

                    step += 1

            except KeyboardInterrupt:
                logging.info('Interrupted')
                coord.request_stop()
            except Exception as e:
                logging.info('Exception')
                coord.request_stop(e)
            finally:
                save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                logging.info("Model saved in file: %s" % save_path)
                # When done, ask the threads to stop.
                coord.request_stop()
                coord.join(threads)

