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
from network import pyramid_processing, pyramid_processing_bidirection, Generator, Discriminator, OpticalFlow
from layer import get_shape
from datasets import BasicDataset, CycleFlow_Dataset
from utils import average_gradients, lrelu, occlusion, rgb_bgr, flow_warp_error, ImagePool
import utils
from data_augmentation import flow_resize
from flowlib import flow_to_color, write_flo
from warp import tf_warp
from loss import epe_loss, abs_robust_loss, census_loss, discriminator_loss, generator_loss, cycle_consistency_loss, flow_consistency
import logging

""" 
description
    building model
    create data of one iterator, loss, train strategy, test stage
 """
class CycleFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5, 
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False, 
                 regularizer_scale=1e-4, lambda1=10, lambda2=10, ngf=64, pool_size=50, use_lsgan=True,
                 cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints', 
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_distillation", training_stage="first_stage",
                 is_restore_model=False, flow_restore_model='./models/KITTI/no_census_no_occlusion', cyclegan_restore_model='./models/KITTI/data_cyclegan',
                 total_restore_model='./KITTI/models/total_model',
                 dataset_config={}, distillation_config={}):


        # optical flow param
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
        self.flow_restore_model = flow_restore_model
        self.cyclegan_restore_model = cyclegan_restore_model
        self.total_restore_model = total_restore_model
        self.dataset_config = dataset_config
        self.distillation_config = distillation_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))

        regularizer_ = slim.l2_regularizer(scale=self.regularizer_scale)
        self.trainable = True
        self.reuse = None
        self.regularizer = regularizer_
        self.is_scale = True
        self.is_bidirection = True

        # CycleGan param
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        use_sigmoid = not use_lsgan
        self.norm = 'instance'  #instance or batch
        self.ngf = ngf
        self.pool_size = pool_size
        # 二阶段训练必须保持 crop_w和crop_h一致, 256x256
        self.image_size = self.dataset_config['crop_w']
        
        self.is_training = True

        self.FlowNet_C1 = OpticalFlow(train=self.is_training, trainable=self.trainable, reuse=self.reuse, regularizer=self.regularizer,
                                is_scale=self.is_scale, is_bidirection=self.is_bidirection, name="clean_flow")
            
        self.FlowNet_C2 = OpticalFlow(train=False, trainable=False, reuse=self.reuse, regularizer=None,
                                is_scale=self.is_scale, is_bidirection=self.is_bidirection, name="clean_flow")
        self.FlowNet_D2 = OpticalFlow(train=self.is_training, trainable=self.trainable, reuse=self.reuse, regularizer=self.regularizer,
                                is_scale=self.is_scale, is_bidirection=self.is_bidirection, name="derain_flow")
        self.FlowNet_R2 = OpticalFlow(train=self.is_training, trainable=self.trainable, reuse=self.reuse, regularizer=self.regularizer,
                                is_scale=self.is_scale, is_bidirection=self.is_bidirection, name="rain_flow")

        self.G = Generator('CycleGan/G', self.is_training, ngf=self.ngf, norm=self.norm, image_size=self.image_size)
        self.D_Y = Discriminator('CycleGan/D_Y',
            self.is_training, norm=self.norm, use_sigmoid=use_sigmoid)
        self.F = Generator('CycleGan/F', self.is_training, norm=self.norm, image_size=self.image_size)
        self.D_X = Discriminator('CycleGan/D_X',
            self.is_training, norm=self.norm, use_sigmoid=use_sigmoid)

        # self.fake_x = tf.placeholder(tf.float32,
        #     shape=[batch_size, image_size, image_size, 3])
        # self.fake_y = tf.placeholder(tf.float32,
        #     shape=[batch_size, image_size, image_size, 3])
        
        # 待定修改
        # if self.load_model is not None:
        #     self.checkpoints_dir = "checkpoints/" + self.load_model.lstrip("checkpoints/")
        # else:
        #     current_time = datetime.now().strftime("%Y%m%d-%H%M")
        #     self.checkpoints_dir = "checkpoints/{}".format(current_time)
        #     try:
        #         os.makedirs(self.checkpoints_dir)
        #     except os.error:
        #         pass

        
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

    def create_dataset_and_iterator(self, training_stage='first_stage', training_mode='no_distillation'):
        if training_stage == 'first_stage':
            if training_mode=='no_distillation':
                dataset = CycleFlow_Dataset(crop_h=self.dataset_config['crop_h'], 
                                    crop_w=self.dataset_config['crop_w'],
                                    batch_size=self.batch_size_per_gpu,
                                    data_list_file=self.dataset_config['x_data_list_file'],
                                    img_dir=self.dataset_config['img_dir'], 
                            fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
                iterator = dataset.create_batch_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                    shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)   
            elif training_mode == 'distillation':
                dataset = CycleFlow_Dataset(crop_h=self.dataset_config['crop_h'], 
                                    crop_w=self.dataset_config['crop_w'],
                                    batch_size=self.batch_size_per_gpu,
                                    data_list_file=self.dataset_config['x_data_list_file'],
                                    img_dir=self.dataset_config['img_dir'],
                            fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
                iterator = dataset.create_batch_distillation_iterator(data_list=dataset.data_list, batch_size=dataset.batch_size,
                    shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads) 
            else:
                raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')
            return dataset, iterator
        elif training_stage == 'second_stage':
            x_dataset = CycleFlow_Dataset(crop_h=self.dataset_config['crop_h'], 
                            crop_w=self.dataset_config['crop_w'],
                            batch_size=self.batch_size_per_gpu,
                            data_list_file=self.dataset_config['x_data_list_file'],
                            img_dir=self.dataset_config['img_dir'],
                    fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
                            
            x_iterator = x_dataset.create_batch_iterator(data_list=x_dataset.data_list, batch_size=x_dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)  

            y_dataset = CycleFlow_Dataset(crop_h=self.dataset_config['crop_h'], 
                                crop_w=self.dataset_config['crop_w'],
                                batch_size=self.batch_size_per_gpu,
                                data_list_file=self.dataset_config['y_data_list_file'],
                                img_dir=self.dataset_config['img_dir'],
                    fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
            y_iterator = y_dataset.create_batch_iterator(data_list=y_dataset.data_list, batch_size=y_dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)    
            return x_dataset, x_iterator, y_dataset, y_iterator
        
        elif training_stage == 'third_stage':
            x_dataset = CycleFlow_Dataset(crop_h=self.dataset_config['crop_h'], 
                            crop_w=self.dataset_config['crop_w'],
                            batch_size=self.batch_size_per_gpu,
                            data_list_file=self.dataset_config['x_data_list_file'],
                            img_dir=self.dataset_config['img_dir'],
                    fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
                            
            x_iterator = x_dataset.create_batch_iterator(data_list=x_dataset.data_list, batch_size=x_dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)  

            y_dataset = CycleFlow_Dataset(crop_h=self.dataset_config['crop_h'], 
                                crop_w=self.dataset_config['crop_w'],
                                batch_size=self.batch_size_per_gpu,
                                data_list_file=self.dataset_config['y_data_list_file'],
                                img_dir=self.dataset_config['img_dir'],
                    fake_flow_occ_dir=self.distillation_config['fake_flow_occ_dir'])
            y_iterator = y_dataset.create_batch_iterator(data_list=y_dataset.data_list, batch_size=y_dataset.batch_size,
                shuffle=True, buffer_size=self.buffer_size, num_parallel_calls=self.num_input_threads)    
            
            return x_dataset, x_iterator, y_dataset, y_iterator


    def compute_flow_losses(self, batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=True, is_scale=True):
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
        
        return losses
    

    def add_loss_summary(self, losses, keys=['abs_robust_mean'], prefix=None):
        for key in keys:
            for loss_key, loss_value in losses[key].items():
                if prefix:
                    loss_name = '%s/%s/%s' % (prefix, key, loss_key)
                else:
                    loss_name = '%s/%s' % (key, loss_key)
                tf.summary.scalar(loss_name, loss_value)



    def model_first_stage(self, iterator, training_mode='no_distillation'):
        if training_mode == 'no_distillation':
            batch_img1, batch_img2 = iterator.get_next()
            flow_fw, flow_bw = self.FlowNet_C1([batch_img1, batch_img2])

            occ_fw, occ_bw = occlusion(flow_fw['full_res'], flow_bw['full_res'])
            mask_fw = 1. - occ_fw
            mask_bw = 1. - occ_bw  

            losses = self.compute_flow_losses(batch_img1, batch_img2, flow_fw, flow_bw, mask_fw, mask_bw, train=self.is_training, is_scale=self.is_scale)
            
            l2_regularizer = tf.losses.get_regularization_losses()
            regularizer_loss = tf.add_n(l2_regularizer) 
        elif training_mode == 'distillation':
            batch_img1, batch_img2, flow_fw, flow_bw, occ_fw, occ_bw = iterator.get_next()
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
            
            flow_fw_patch, flow_bw_patch = self.FlowNet_C1([batch_img1_cropped_patch, batch_img2_cropped_patch])

            occ_fw_patch, occ_bw_patch = occlusion(flow_fw_patch['full_res'], flow_bw_patch['full_res'])
            mask_fw_patch = 1. - occ_fw_patch
            mask_bw_patch = 1. - occ_bw_patch

            losses = self.compute_flow_losses(batch_img1_cropped_patch, batch_img2_cropped_patch, flow_fw_patch, flow_bw_patch, mask_fw_patch, mask_bw_patch, train=self.is_training, is_scale=self.is_scale)

            valid_mask_fw = tf.clip_by_value(occ_fw_patch - occ_fw_cropped_patch, 0., 1.)
            valid_mask_bw = tf.clip_by_value(occ_bw_patch - occ_bw_cropped_patch, 0., 1.)
            data_distillation_loss = {}
            data_distillation_loss['distillation'] = (abs_robust_loss(flow_fw_cropped_patch-flow_fw_patch['full_res'], valid_mask_fw) + \
                                        abs_robust_loss(flow_bw_cropped_patch-flow_bw_patch['full_res'], valid_mask_bw)) / 2
            losses['data_distillation'] = data_distillation_loss

            l2_regularizer = tf.losses.get_regularization_losses()
            regularizer_loss = tf.add_n(l2_regularizer)
        else:
            raise ValueError('Invalid training_mode. Training_mode should be one of {no_distillation, distillation}')    
        
        return losses, regularizer_loss

    def model_second_stage(self, x_iterator, y_iterator):
        self.fake_x = tf.placeholder(tf.float32,
        shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.fake_y = tf.placeholder(tf.float32,
        shape=[self.batch_size, self.image_size, self.image_size, 3])
        
        
        x, x_ = x_iterator.get_next()
        y, y_ = y_iterator.get_next()

        # cycle的预处理范围在-1-1
        x = utils.convert2float(x*255)
        y = utils.convert2float(y*255) 

        tf.summary.image('X/X_input', utils.convert2int(x))   
        tf.summary.image('Y/Y_input', utils.convert2int(y))   
        
        

        # image Cycle consistency
        cycle_loss = cycle_consistency_loss(self.G, self.F, x, y, self.lambda1, self.lambda2)

        # X -> Y
        # image
        fake_y = self.G(x)
        G_gan_loss = generator_loss(self.D_Y, fake_y, use_lsgan=self.use_lsgan)
        G_loss =  G_gan_loss + cycle_loss
        D_Y_loss = discriminator_loss(self.D_Y, y, self.fake_y, use_lsgan=self.use_lsgan)
       
        # Y -> X
        # image
        fake_x = self.F(y)
        F_gan_loss = generator_loss(self.D_X, fake_x, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_X_loss = discriminator_loss(self.D_X, x, self.fake_x, use_lsgan=self.use_lsgan)
        

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

        tf.summary.image('X/generated', utils.convert2int(self.G(x)))
        tf.summary.image('X/reconstruction', utils.convert2int(self.F(self.G(x))))
        tf.summary.image('Y/generated', utils.convert2int(self.F(y)))
        tf.summary.image('Y/reconstruction', utils.convert2int(self.G(self.F(y))))

        # 修改返回参数
        return G_loss, D_Y_loss, F_loss, D_X_loss, fake_y, fake_x


    def model_third_stage(self, x_iterator, y_iterator):
        self.fake_x1 = tf.placeholder(tf.float32,
        shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.fake_x2 = tf.placeholder(tf.float32,
        shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.fake_y1 = tf.placeholder(tf.float32,
        shape=[self.batch_size, self.image_size, self.image_size, 3])
        self.fake_y2 = tf.placeholder(tf.float32,
        shape=[self.batch_size, self.image_size, self.image_size, 3])
        
        x1, x2 = x_iterator.get_next()
        y1, y2 = y_iterator.get_next()

        tf.summary.image('X/X_input', x1)   
        tf.summary.image('Y/Y_input', y1)   

        # convert to pre-process of cyclegan
        x1_img = utils.convert2float(x1*255)
        x2_img = utils.convert2float(x2*255)
        y1_img = utils.convert2float(y1*255)
        y2_img = utils.convert2float(y2*255)
        
        
        # image Cycle consistency
        cycle_loss_1 = cycle_consistency_loss(self.G, self.F, x1_img, y1_img, self.lambda1, self.lambda2)
        cycle_loss_2 = cycle_consistency_loss(self.G, self.F, x2_img, y2_img, self.lambda1, self.lambda2)
        cycle_loss = cycle_loss_1 + cycle_loss_2

        # X -> Y
        # flow
        flow_clean_x_f, flow_clean_x_b = self.FlowNet_C2([x1, x2])
        occ_fw_x, occ_bw_x = occlusion(flow_clean_x_f['full_res'], flow_clean_x_b['full_res'])
        mask_fw_x = 1. - occ_fw_x
        mask_bw_x = 1. - occ_bw_x
        # image
        fake_y1 = self.G(x1_img)
        fake_y2 = self.G(x2_img)
        G_gan_loss_1 = generator_loss(self.D_Y, fake_y1, use_lsgan=self.use_lsgan)
        G_gan_loss_2 = generator_loss(self.D_Y, fake_y2, use_lsgan=self.use_lsgan)
        G_gan_loss = G_gan_loss_1 + G_gan_loss_2
        G_loss =  G_gan_loss + cycle_loss
        D_Y_loss_1 = discriminator_loss(self.D_Y, y1_img, self.fake_y1, use_lsgan=self.use_lsgan)
        D_Y_loss_2 = discriminator_loss(self.D_Y, y2_img, self.fake_y2, use_lsgan=self.use_lsgan)
        D_Y_loss = D_Y_loss_1 + D_Y_loss_2

        # Fake Y -> X 
        # flow
        # flow_fake_y2 = utils.convert2int(fake_y2)/255.   (fake_y1+1.0)/2.0
        flow_rain_x_f, flow_rain_x_b = self.FlowNet_R2([(fake_y1+1.0)/2.0, (fake_y2+1.0)/2.0])
        cycle_x1 = self.F(fake_y1)
        cycle_x2 = self.F(fake_y2)
        flow_derain_x_f, flow_derain_x_b = self.FlowNet_D2([(cycle_x1+1.0)/2.0, (cycle_x2+1.0)/2.0])

        flow_consistency_CR = flow_consistency(flow_clean_x_f['full_res'], flow_rain_x_f['full_res'], tf.ones_like(mask_fw_x)) * 0.8
        flow_consistency_CD = flow_consistency(flow_clean_x_f['full_res'], flow_derain_x_f['full_res'], tf.ones_like(mask_fw_x)) * 0.8


        # Y -> X
        # flow
        flow_T_rain_y_f, flow_T_rain_y_b = self.FlowNet_R2([y1, y2]) # Real Rain flow
        
        # image
        fake_x1 = self.F(y1_img)
        fake_x2 = self.F(y2_img)
        F_gan_loss_1 = generator_loss(self.D_X, fake_x1, use_lsgan=self.use_lsgan)
        F_gan_loss_2 = generator_loss(self.D_X, fake_x2, use_lsgan=self.use_lsgan)
        F_gan_loss = F_gan_loss_1 + F_gan_loss_2
        F_loss = F_gan_loss + cycle_loss
        D_X_loss_1 = discriminator_loss(self.D_X, x1_img, self.fake_x1, use_lsgan=self.use_lsgan)
        D_X_loss_2 = discriminator_loss(self.D_X, x2_img, self.fake_x2, use_lsgan=self.use_lsgan)
        D_X_loss = D_X_loss_1 + D_X_loss_2

        # Fake X -> Y
        # flow  (fake_x1+1.0)/2.0,
        flow_derain_y_f, flow_derain_y_b = self.FlowNet_D2([(fake_x1+1.0)/2.0, (fake_x2+1.0)/2.0])
        occ_fw_y, occ_bw_y = occlusion(flow_derain_y_f['full_res'], flow_derain_y_b['full_res'])
        mask_fw_y = 1. - occ_fw_y
        mask_bw_y = 1. - occ_bw_y

        cycle_y1 = self.F(fake_x1)
        cycle_y2 = self.F(fake_x2)
        flow_rain_y_f, flow_rain_y_b = self.FlowNet_R2([(cycle_y1+1.0)/2.0, (cycle_y2+1.0)/2.0])  # Fake Rain flow

        flow_consistency_DTR = flow_consistency(flow_derain_y_f['full_res'], flow_T_rain_y_f['full_res'], tf.ones_like(mask_fw_y))
        flow_consistency_DR = flow_consistency(flow_derain_y_f['full_res'], flow_rain_y_f['full_res'], tf.ones_like(mask_fw_y))
        flow_consistency_Y2X2Y = (flow_consistency_DTR + flow_consistency_DR) / 2.0

        # summary
        tf.summary.histogram('D_Y/true', self.D_Y(y1))
        tf.summary.histogram('D_Y/fake', self.D_Y(self.G(x1)))
        tf.summary.histogram('D_X/true', self.D_X(x1))
        tf.summary.histogram('D_X/fake', self.D_X(self.F(y1)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_Y', D_Y_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_X', D_X_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)
        tf.summary.scalar('loss/flow_X', flow_consistency_CD)
        tf.summary.scalar('loss/flow_Y', flow_consistency_DTR)

        # tf.summary.image('X/generated', self.G(x1))
        # tf.summary.image('X/reconstruction', self.F(self.G(x1)))
        # tf.summary.image('Y/generated', self.F(y1))
        # tf.summary.image('Y/reconstruction', self.G(self.F(y1)))

        tf.summary.image('X/generated', utils.convert2int(self.G(x1_img)))
        tf.summary.image('X/reconstruction', utils.convert2int(self.F(self.G(x1_img))))
        tf.summary.image('Y/generated', utils.convert2int(self.F(y1_img)))
        tf.summary.image('Y/reconstruction', utils.convert2int(self.G(self.F(y1_img))))
        flow_est_color = flow_to_color(flow_clean_x_f['full_res'], mask=None, max_flow=256)
        tf.summary.image('Flow/clean', utils.convert2int(flow_est_color))
        flow_est_color_rain = flow_to_color(flow_rain_x_f['full_res'], mask=None, max_flow=256)
        tf.summary.image('Flow/rain', utils.convert2int(flow_est_color_rain))
        flow_est_color_derain = flow_to_color(flow_derain_x_f['full_res'], mask=None, max_flow=256)
        tf.summary.image('Flow/derain', utils.convert2int(flow_est_color_derain))
        

        # 修改返回参数
        return flow_consistency_CR, flow_consistency_CD, flow_consistency_Y2X2Y, G_loss, D_Y_loss, F_loss, D_X_loss, fake_y1, fake_y2, fake_x1, fake_x2
    
    def optimize(self, losses, training_stage='second_stage'):
        def make_optimizer(loss, variables, name='Adam'):
            """ Adam optimizer with learning rate 0.0002 for the first 100k steps (~100 epochs)
                and a linearly decaying rate that goes to zero over the next 100k steps
            """
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.initial_learning_rate
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
        # 优化顺序待商讨和训练策略
        G_loss = losses['G_loss']
        D_Y_loss = losses['D_Y_loss']
        F_loss = losses['F_loss']
        D_X_loss = losses['D_X_loss']

        if training_stage == 'third_stage':
            flow_consistency_CR = losses['flow_CR']
            flow_consistency_CD = losses['flow_CD']
            flow_consistency_Y2X2Y = losses['flow_Y2X2Y']

        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_Y_optimizer = make_optimizer(D_Y_loss, self.D_Y.variables, name='Adam_D_Y')
        F_optimizer =  make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_X_optimizer = make_optimizer(D_X_loss, self.D_X.variables, name='Adam_D_X')

        if training_stage == 'third_stage':
            # 光流一致性训练函数是否还可以进一步优化
            Flow_CR_optimizer = make_optimizer(flow_consistency_CR, self.FlowNet_R2.variables, name='Adam_Flow_RainX')
            Flow_CD_optimizer = make_optimizer(flow_consistency_CD, self.FlowNet_D2.variables, name='Adam_Flow_DerainX')
            # Flow_Y2X2Y_optimizer = make_optimizer(flow_consistency_Y2X2Y, self.FlowNet_R2.variables, name='Adam_Flow_RainY')

            with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer, Flow_CR_optimizer, Flow_CD_optimizer]):
                return tf.no_op(name='optimizers')
        elif training_stage == 'second_stage':
            with tf.control_dependencies([G_optimizer, D_Y_optimizer, F_optimizer, D_X_optimizer]):
                return tf.no_op(name='optimizers')

    def train(self, training_stage='first_stage'):
        if training_stage == 'first_stage':    
            with tf.Graph().as_default(), tf.device(self.shared_device):
                self.global_step = tf.Variable(0, trainable=False)
                self.dataset, self.iterator = self.create_dataset_and_iterator(training_stage=training_stage, training_mode=self.training_mode)
                self.lr_decay = tf.train.exponential_decay(self.initial_learning_rate, self.global_step, decay_steps=self.decay_steps, decay_rate=self.decay_rate, staircase=True)
                tf.summary.scalar('learning_rate', self.lr_decay)
                self.optim = tf.train.AdamOptimizer(self.lr_decay, self.beta1) 
                self.losses, self.regularizer_loss = self.model_first_stage(self.iterator, self.training_mode)

                # training mode select
                if self.training_mode == 'no_distillation':
                    self.optim_loss = self.losses['abs_robust_mean']['no_occlusion']
                elif self.training_mode == 'distillation':
                    self.optim_loss = self.losses['census']['occlusion'] + self.losses['data_distillation']['distillation']
                
                
                self.train_op = self.optim.minimize(self.optim_loss, var_list=tf.trainable_variables(), global_step=self.global_step)

                self.add_loss_summary(self.losses, keys=self.losses.keys())
                tf.summary.scalar('regularizer_loss', self.regularizer_loss)

                merge_summary = tf.summary.merge_all()
                summary_writer = tf.summary.FileWriter(logdir='/'.join([self.summary_dir, 'train', self.model_name]))
                self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
                self.saver = tf.train.Saver(var_list=self.trainable_vars + [self.global_step], max_to_keep=500)

                self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=self.allow_soft_placement, log_device_placement=self.log_device_placement))

                self.sess.run(tf.global_variables_initializer())
        
                self.sess.run(tf.local_variables_initializer())

                if self.is_restore_model:
                    self.saver.restore(self.sess, self.flow_restore_model)
                
                self.sess.run(tf.assign(self.global_step, 0))
                start_step = self.sess.run(self.global_step)
                self.sess.run(self.iterator.initializer)

                start_time = time.time()

                for step in range(start_step+1, self.iter_steps+1):
                    _, abs_robust_mean_no_occlusion, census_occlusion = self.sess.run([self.train_op,
                        self.losses['abs_robust_mean']['no_occlusion'], self.losses['abs_robust_mean']['occlusion']])
                    if np.mod(step, self.display_log_interval) == 0:
                        print('step: %d time: %.6fs, abs_robust_mean_no_occlusion: %.6f, census_occlusion: %.6f' % 
                            (step, time.time() - start_time, abs_robust_mean_no_occlusion, census_occlusion)) 
                
                    
                    if np.mod(step, self.write_summary_interval) == 0:
                        summary_str = self.sess.run(merge_summary)
                        summary_writer.add_summary(summary_str, global_step=step)
                    
                    if np.mod(step, self.save_checkpoint_interval) == 0:
                        self.saver.save(self.sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                        write_meta_graph=False, write_state=False) 
        elif training_stage == 'second_stage':   
            print("second stage training...") 

            # 此处添加二阶段训练
            graph = tf.Graph()
            with graph.as_default():
                self.global_step = tf.Variable(0, trainable=False)
                self.X_dataset, self.X_iterator, self.Y_dataset, self.Y_iterator = self.create_dataset_and_iterator(training_stage=training_stage, training_mode=self.training_mode)

                # build model 
                losses = {}
                losses['G_loss'], losses['D_Y_loss'], losses['F_loss'], losses['D_X_loss'], \
                fake_y, fake_x = self.model_second_stage(self.X_iterator, self.Y_iterator)
                optimizers = self.optimize(losses, training_stage=training_stage)

                summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(self.checkpoint_dir, graph)
                

                # 导入特定参数
                self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
                self.save_saver = tf.train.Saver(var_list=self.trainable_vars + [self.global_step], max_to_keep=500)
       

            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                step = 0
                # 需要做的就是 如何把原模型加载进来
                if self.is_restore_model:
                    self.save_saver.restore(sess, self.cyclegan_restore_model)
                    

                    # checkpoint = tf.train.get_checkpoint_state(self.checkpoints_dir)
                    # meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    # restore = tf.train.import_meta_graph(meta_graph_path)
                    # restore.restore(sess, tf.train.latest_checkpoint(self.checkpoints_dir))
                    # step = int(meta_graph_path.split("-")[2].split(".")[0])


                # else:
                #     sess.run(tf.global_variables_initializer())
                #     sess.run(tf.local_variables_initializer())
                #     step = 0

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
                                [optimizers, losses['G_loss'], losses['D_Y_loss'], losses['F_loss'], losses['D_X_loss'], summary_op],
                                feed_dict={self.fake_y: fake_Y_pool.query(fake_y_val),   
                                            self.fake_x: fake_X_pool.query(fake_x_val)}
                            )
                        )

                        if step % self.write_summary_interval == 0:
                            logging.info('-----------Step %d:-------------' % step)
                            logging.info('  G_loss   : {}'.format(G_loss_val))
                            logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                            logging.info('  F_loss   : {}'.format(F_loss_val))
                            logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                            train_writer.add_summary(summary, step)
                            train_writer.flush()
                        
                        if step % self.save_checkpoint_interval == 0:
                            # save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                            save_path = self.save_saver.save(sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                        write_meta_graph=False, write_state=False) 
                            logging.info("Model saved in file: %s" % save_path)
                        
                        step += 1


                except KeyboardInterrupt:
                    logging.info('Interrupted')
                    coord.request_stop()
                except Exception as e:
                    logging.info('Exception')
                    coord.request_stop(e)
                finally:
                    # save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                    save_path = self.save_saver.save(sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                        write_meta_graph=False, write_state=False) 
                    logging.info("Model saved in file: %s" % save_path)
                    # When done, ask the threads to stop.
                    coord.request_stop()
                    coord.join(threads)


        elif training_stage == 'third_stage':   
            print("third stage training...") 
            # 此处添加三阶段训练
            graph = tf.Graph()
            with graph.as_default():
                self.global_step = tf.Variable(0, trainable=False)
                self.X_dataset, self.X_iterator, self.Y_dataset, self.Y_iterator = self.create_dataset_and_iterator(training_stage=training_stage, training_mode=self.training_mode)

                # build model 
                losses = {}
                losses['flow_CR'], losses['flow_CD'], losses['flow_Y2X2Y'], losses['G_loss'], \
                losses['D_Y_loss'], losses['F_loss'], losses['D_X_loss'], \
                fake_y1, fake_y2, fake_x1, fake_x2 = self.model_third_stage(self.X_iterator, self.Y_iterator)
                optimizers = self.optimize(losses, training_stage=training_stage)

                summary_op = tf.summary.merge_all()
                train_writer = tf.summary.FileWriter(self.checkpoint_dir, graph)
                

                # 导入特定参数
                self.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) 
                var = tf.global_variables()
                var_flow_restore = [val for val in var if 'clean_flow' in val.name]
                self.flow_saver = tf.train.Saver(var_flow_restore)
                
                var_cyclegan_restore = [val for val in self.trainable_vars if 'CycleGan' in val.name]
                self.cyclegan_saver = tf.train.Saver(var_cyclegan_restore)

                self.save_saver = tf.train.Saver(var_list=var_flow_restore + self.trainable_vars + [self.global_step], max_to_keep=500)
                
                

            with tf.Session(graph=graph, config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                step = 0
                # 需要做的就是 如何把原模型加载进来
                
                if self.is_restore_model:
                    self.save_saver.restore(sess, self.total_restore_model)
                else:
                    self.flow_saver.restore(sess, self.flow_restore_model)
                    self.cyclegan_saver.restore(sess, self.cyclegan_restore_model)

                    # checkpoint = tf.train.get_checkpoint_state(self.checkpoints_dir)
                    # meta_graph_path = checkpoint.model_checkpoint_path + ".meta"
                    # restore = tf.train.import_meta_graph(meta_graph_path)
                    # restore.restore(sess, tf.train.latest_checkpoint(self.checkpoints_dir))
                    # step = int(meta_graph_path.split("-")[2].split(".")[0])


                # else:
                #     sess.run(tf.global_variables_initializer())
                #     sess.run(tf.local_variables_initializer())
                #     step = 0

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
                        fake_y1_val, fake_y2_val, fake_x1_val, fake_x2_val = sess.run([fake_y1, fake_y2, fake_x1, fake_x2])

                        # train
                        _, G_loss_val, D_Y_loss_val, F_loss_val, D_X_loss_val, Flow_Consistency_CR_val, Flow_Consistency_CD_val, summary = (
                            sess.run(
                                [optimizers, losses['G_loss'], losses['D_Y_loss'], losses['F_loss'], losses['D_X_loss'], \
                                losses['flow_CR'], losses['flow_CD'], summary_op],
                                feed_dict={self.fake_y1: fake_Y_pool.query(fake_y1_val),
                                            self.fake_y2: fake_Y_pool.query(fake_y2_val),
                                            self.fake_x1: fake_X_pool.query(fake_x1_val),
                                            self.fake_x2: fake_X_pool.query(fake_x2_val)}
                            )
                        )

                        if step % self.write_summary_interval == 0:
                            logging.info('-----------Step %d:-------------' % step)
                            logging.info('  G_loss   : {}'.format(G_loss_val))
                            logging.info('  D_Y_loss : {}'.format(D_Y_loss_val))
                            logging.info('  F_loss   : {}'.format(F_loss_val))
                            logging.info('  D_X_loss : {}'.format(D_X_loss_val))
                            logging.info('  Flow_CR_loss : {}'.format(Flow_Consistency_CR_val))
                            logging.info('  Flow_CD_loss : {}'.format(Flow_Consistency_CD_val))
                            train_writer.add_summary(summary, step)
                            train_writer.flush()
                        
                        if step % self.save_checkpoint_interval == 0:
                            # save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                            save_path = self.save_saver.save(sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                        write_meta_graph=False, write_state=False) 
                            logging.info("Model saved in file: %s" % save_path)
                        
                        step += 1


                except KeyboardInterrupt:
                    logging.info('Interrupted')
                    coord.request_stop()
                except Exception as e:
                    logging.info('Exception')
                    coord.request_stop(e)
                finally:
                    # save_path = saver.save(sess, self.checkpoints_dir + "/model.ckpt", global_step=step)
                    save_path = self.save_saver.save(sess, '/'.join([self.checkpoint_dir, self.model_name, 'model']), global_step=step, 
                                        write_meta_graph=False, write_state=False) 
                    logging.info("Model saved in file: %s" % save_path)
                    # When done, ask the threads to stop.
                    coord.request_stop()
                    coord.join(threads)
        
        else:
            raise ValueError('Invalid training_stage. Training_mode should be one of {first_stage, second_stage, third_stage}')    




    def test(self, restore_model, save_dir):
        print("test demo")
        flownet = OpticalFlow(train=False, trainable=False, reuse=None, regularizer=None, is_scale=True, is_bidirection=False, name='clean_flow')
        dataset = CycleFlow_Dataset(data_list_file=self.dataset_config['x_data_list_file'], img_dir=self.dataset_config['img_dir'])
        save_name_list = dataset.data_list[:, 2]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img1, batch_img2 = iterator.get_next()
        
        
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

    
    def generate_fake_flow_occlusion(self, restore_model, save_dir, training_stage='first_stage'):
        if training_stage == 'first_stage':    
            flownet = OpticalFlow(train=False, trainable=False, reuse=None, regularizer=None, is_scale=True, is_bidirection=True, name='clean_flow')

            dataset = CycleFlow_Dataset(data_list_file=self.dataset_config['x_data_list_file'], img_dir=self.dataset_config['img_dir'])
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
        else:
            raise ValueError('Invalid training_stage. training_stage should be only in first_stage')    



        
