# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc as misc
import cv2
import matplotlib.pyplot as plt
from flowlib import read_flo, read_pfm, read_vkitti_png_flow
from data_augmentation import *
from utils import imshow   

""" 
description
    building dataset class including read, decode(image), data pre-process, batch distribute
 """


class BasicDataset(object):
    def __init__(self, crop_h=320, crop_w=896, batch_size=4, data_list_file='path_to_your_data_list_file', 
                 img_dir='path_to_your_image_directory', fake_flow_occ_dir='path_to_your_fake_flow_occlusion_directory'):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.data_list = np.loadtxt(data_list_file, dtype=np.str)
        self.data_num = self.data_list.shape[0]
        self.fake_flow_occ_dir = fake_flow_occ_dir
    
    # KITTI's data format for storing flow and mask
    # The first two channels are flow, the third channel is mask
    def extract_flow_and_mask(self, flow):
        optical_flow = flow[:, :, :2]
        optical_flow = (optical_flow - 32768) / 64.0
        mask = tf.cast(tf.greater(flow[:, :, 2], 0), tf.float32)
        #mask = tf.cast(flow[:, :, 2], tf.float32)
        mask = tf.expand_dims(mask, -1)
        return optical_flow, mask    
    
    # The default image type is PNG.
    def read_and_decode(self, filename_queue):
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    
        return img1, img2 

    # For Flying Chairs, the image type is ppm, please use "read_and_decode_ppm" instead of "read_and_decode".
    # Similarily, for other image types, please write their decode functions by yourself.
    def read_and_decode_ppm(self, filename_queue):
        def read_ppm(self, filename):
            img = misc.imread(filename).astype('float32')
            return img   
        
        flying_h = 384
        flying_w = 512
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])

        img1 = tf.py_func(read_ppm, [img1_name], tf.float32)
        img2 = tf.py_func(read_ppm, [img2_name], tf.float32)

        img1 = tf.reshape(img1, [flying_h, flying_w, 3])
        img2 = tf.reshape(img2, [flying_h, flying_w, 3])
        return img1, img2       
    # VIKKI image-flow
    def read_and_decode_VKITTI(self, filename_queue):
        rgb_img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        rgb_img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])
        rain_img1_name = tf.string_join([self.img_dir, '/', filename_queue[2]])
        rain_img2_name = tf.string_join([self.img_dir, '/', filename_queue[3]])
        derain_img1_name = tf.string_join([self.img_dir, '/', filename_queue[4]])
        derain_img2_name = tf.string_join([self.img_dir, '/', filename_queue[5]])
       
        flow_name = tf.string_join([self.img_dir, '/', filename_queue[6]])
        

        rgb_img1 = tf.image.decode_png(tf.read_file(rgb_img1_name), channels=3)
        rgb_img1 = tf.cast(rgb_img1, tf.float32)
        rgb_img2 = tf.image.decode_png(tf.read_file(rgb_img2_name), channels=3)
        rgb_img2 = tf.cast(rgb_img2, tf.float32) 
        rain_img1 = tf.image.decode_png(tf.read_file(rain_img1_name), channels=3)
        rain_img1 = tf.cast(rain_img1, tf.float32)
        rain_img2 = tf.image.decode_png(tf.read_file(rain_img2_name), channels=3)
        rain_img2 = tf.cast(rain_img2, tf.float32) 
        derain_img1 = tf.image.decode_png(tf.read_file(derain_img1_name), channels=3)
        derain_img1 = tf.cast(derain_img1, tf.float32)
        derain_img2 = tf.image.decode_png(tf.read_file(derain_img2_name), channels=3)
        derain_img2 = tf.cast(derain_img2, tf.float32) 
                
        # tf-string to string

        flow = tf.py_func(read_vkitti_png_flow, [flow_name], tf.float32)
  
        flow = tf.convert_to_tensor(flow, tf.float32)    
        flow = tf.cast(flow, tf.float32)

        return rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, flow

    def read_and_decode_distillation(self, filename_queue):
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])     
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    
        
        flow_occ_fw_name = tf.string_join([self.fake_flow_occ_dir, '/flow_occ_fw_', filename_queue[2], '.png'])
        flow_occ_bw_name = tf.string_join([self.fake_flow_occ_dir, '/flow_occ_bw_', filename_queue[2], '.png'])
        flow_occ_fw = tf.image.decode_png(tf.read_file(flow_occ_fw_name), dtype=tf.uint16, channels=3)
        flow_occ_fw = tf.cast(flow_occ_fw, tf.float32)   
        flow_occ_bw = tf.image.decode_png(tf.read_file(flow_occ_bw_name), dtype=tf.uint16, channels=3)
        flow_occ_bw = tf.cast(flow_occ_bw, tf.float32)             
        flow_fw, occ_fw = self.extract_flow_and_mask(flow_occ_fw)
        flow_bw, occ_bw = self.extract_flow_and_mask(flow_occ_bw)
        return img1, img2, flow_fw, flow_bw, occ_fw, occ_bw  

    def augmentation(self, img1, img2):
        img1, img2 = random_crop([img1, img2], self.crop_h, self.crop_w)
        img1, img2 = random_flip([img1, img2])
        img1, img2 = random_channel_swap([img1, img2])
        return img1, img2 
    
    def augmentation_supervised(self, rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, img_gt):
        rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, img_gt= random_crop([rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, img_gt], self.crop_h, self.crop_w)
        # img1, img2 = random_flip([img1, img2])
        # img1, img2 = random_channel_swap([img1, img2])
        return rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, img_gt

    def augmentation_distillation(self, img1, img2, flow_fw, flow_bw, occ_fw, occ_bw):
        [img1, img2, flow_fw, flow_bw, occ_fw, occ_bw] = random_crop([img1, img2, flow_fw, flow_bw, occ_fw, occ_bw], self.crop_h, self.crop_w)
        [img1, img2, occ_fw, occ_bw], [flow_fw, flow_bw] = random_flip_with_flow([img1, img2, occ_fw, occ_bw], [flow_fw, flow_bw])
        img1, img2 = random_channel_swap([img1, img2])
        return img1, img2, flow_fw, flow_bw, occ_fw, occ_bw

    def preprocess_augmentation(self, filename_queue):
        img1, img2 = self.read_and_decode(filename_queue)
        img1 = img1 / 255.
        img2 = img2 / 255.        
        img1, img2 = self.augmentation(img1, img2)
        return img1, img2
    
    def preprocess_Supervised_flow_augmentation(self, filename_queue):
        rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, flow = self.read_and_decode_VKITTI(filename_queue)
        
        rgb_img1 = rgb_img1 / 255.
        rgb_img2 = rgb_img2 / 255.     
        rain_img1 = rain_img1 / 255.
        rain_img2 = rain_img2 / 255.   
        derain_img1 = derain_img1 / 255.
        derain_img2 = derain_img2 / 255.   

        rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, flow = self.augmentation_supervised(img1, img2, flow)
        return rgb_img1, rgb_img2, rain_img1, rain_img2, derain_img1, derain_img2, flow

    def preprocess_augmentation_distillation(self, filename_queue):
        img1, img2, flow_fw, flow_bw, occ_fw, occ_bw = self.read_and_decode_distillation(filename_queue)
        img1 = img1 / 255.
        img2 = img2 / 255.        
        img1, img2, flow_fw, flow_bw, occ_fw, occ_bw = self.augmentation_distillation(img1, img2, flow_fw, flow_bw, occ_fw, occ_bw)
        return img1, img2, flow_fw, flow_bw, occ_fw, occ_bw  

    def preprocess_one_shot(self, filename_queue):
        img1, img2 = self.read_and_decode(filename_queue)
        img1 = img1 / 255.
        img2 = img2 / 255.        
        return img1, img2
    
    def create_batch_iterator(self, data_list, batch_size, shuffle=True, buffer_size=5000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator

    def create_batch_distillation_iterator(self, data_list, batch_size, shuffle=True, buffer_size=5000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation_distillation, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator    
    
    def create_one_shot_iterator(self, data_list, num_parallel_calls=4):
        #  For Validation or Testing
        #     Generate image and flow one_by_one without cropping, image and flow size may change every iteration
    
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_one_shot, num_parallel_calls=num_parallel_calls)        
        dataset = dataset.batch(1)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator     
    
    def preprocess_synthetic_one_shot(self, filename_queue):
        def read_img(filename_queue):
            clean_img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
            clean_img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])
            fog_img1_name = tf.string_join([self.img_dir, '/', filename_queue[2]])
            fog_img2_name = tf.string_join([self.img_dir, '/', filename_queue[3]])

            clean_img1 = tf.image.decode_png(tf.read_file(clean_img1_name), channels=3)
            clean_img1 = tf.cast(clean_img1, tf.float32)
            clean_img2 = tf.image.decode_png(tf.read_file(clean_img2_name), channels=3)
            clean_img2 = tf.cast(clean_img2, tf.float32)    
            fog_img1 = tf.image.decode_png(tf.read_file(fog_img1_name), channels=3)
            fog_img1 = tf.cast(fog_img1, tf.float32)
            fog_img2 = tf.image.decode_png(tf.read_file(fog_img2_name), channels=3)
            fog_img2 = tf.cast(fog_img2, tf.float32)    
            return clean_img1, clean_img2, fog_img1, fog_img2
            
        clean_img1, clean_img2, fog_img1, fog_img2 = read_img(filename_queue)
        clean_img1 = clean_img1 / 255.
        clean_img2 = clean_img2 / 255.     
        fog_img1 = fog_img1 / 255.
        fog_img2 = fog_img2 / 255. 
        return clean_img1, clean_img2, fog_img1, fog_img2



# cyclegan-dataset-read
class CycleGan_BasicDataset(object):
    def __init__(self, crop_h=256, crop_w=256, batch_size=1, data_list_file='path_to_your_data_list_file', 
                 img_dir='path_to_your_image_directory'):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.data_list = np.loadtxt(data_list_file, dtype=np.str)
        self.data_num = self.data_list.shape[0]
        
    #     return img 
    def read_and_decode(self, filename_queue):
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])
        img1 = tf.image.decode_jpeg(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_jpeg(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    
        return img1, img2 

    def augmentation(self, img1, img2):
        img1, img2 = random_crop([img1, img2], self.crop_h, self.crop_w)
        # img1, img2 = random_flip([img1, img2])
        # img1, img2 = random_channel_swap([img1, img2])
        return img1, img2 
    
    def augmentation_cycle(self, img1, img2):
        img1 = tf.image.resize_images(img1, size=(256, 256))
        img1.set_shape([256, 256, 3])

        # 区别
        # images = [img1]
        # images = tf.train.shuffle_batch(
        #     [img1], batch_size=self.batch_size, num_threads=4,
        #     capacity=64,
        #     min_after_dequeue=1)

        return img1
    def preprocess_augmentation(self, filename_queue):
        img1, img2 = self.read_and_decode(filename_queue)
        img1 = img1 / 255.       
        img2 = img2 / 255.  

        # 修改此处，修改为适配rain-kitti dataset
        # img1 = self.augmentation_cycle(img1, img2) 
        
        img1, img2 = self.augmentation(img1, img2)
        return img1

    # cyclegan-test
    def create_batch_iterator(self, data_list, batch_size, shuffle=True, buffer_size=5000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator


# cyclegan-dataset-read
class CycleFlow_Dataset(object):
    def __init__(self, crop_h=256, crop_w=256, batch_size=1, data_list_file='path_to_your_data_list_file', 
                 img_dir='path_to_your_image_directory', fake_flow_occ_dir='path_to_your_fake_flow_occlusion_directory'):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.batch_size = batch_size
        self.img_dir = img_dir
        self.data_list = np.loadtxt(data_list_file, dtype=np.str)
        self.data_num = self.data_list.shape[0]
        self.fake_flow_occ_dir = fake_flow_occ_dir
    
    def extract_flow_and_mask(self, flow):
        optical_flow = flow[:, :, :2]
        optical_flow = (optical_flow - 32768) / 64.0
        mask = tf.cast(tf.greater(flow[:, :, 2], 0), tf.float32)
        #mask = tf.cast(flow[:, :, 2], tf.float32)
        mask = tf.expand_dims(mask, -1)
        return optical_flow, mask    

   
    #     return img 
    def read_and_decode(self, filename_queue):
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    
        return img1, img2 

    def read_and_decode_distillation(self, filename_queue):
        img1_name = tf.string_join([self.img_dir, '/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/', filename_queue[1]])     
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    
        
        flow_occ_fw_name = tf.string_join([self.fake_flow_occ_dir, '/flow_occ_fw_', filename_queue[2], '.png'])
        flow_occ_bw_name = tf.string_join([self.fake_flow_occ_dir, '/flow_occ_bw_', filename_queue[2], '.png'])
        flow_occ_fw = tf.image.decode_png(tf.read_file(flow_occ_fw_name), dtype=tf.uint16, channels=3)
        flow_occ_fw = tf.cast(flow_occ_fw, tf.float32)   
        flow_occ_bw = tf.image.decode_png(tf.read_file(flow_occ_bw_name), dtype=tf.uint16, channels=3)
        flow_occ_bw = tf.cast(flow_occ_bw, tf.float32)             
        flow_fw, occ_fw = self.extract_flow_and_mask(flow_occ_fw)
        flow_bw, occ_bw = self.extract_flow_and_mask(flow_occ_bw)
        return img1, img2, flow_fw, flow_bw, occ_fw, occ_bw  

    def read_and_decode_finetune(self, filename_queue):
        img1_name = tf.string_join([self.img_dir, '/image_2/', filename_queue[0]])
        img2_name = tf.string_join([self.img_dir, '/image_2/', filename_queue[1]])     
        img1 = tf.image.decode_png(tf.read_file(img1_name), channels=3)
        img1 = tf.cast(img1, tf.float32)
        img2 = tf.image.decode_png(tf.read_file(img2_name), channels=3)
        img2 = tf.cast(img2, tf.float32)    

        flow_gt_occ_fw_name = tf.string_join([self.img_dir, '/flow_noc/', filename_queue[2], '.png'])
        flow_occ_fw = tf.image.decode_png(tf.read_file(flow_gt_occ_fw_name), dtype=tf.uint16, channels=3)
        flow_occ_fw = tf.cast(flow_occ_fw, tf.float32)   
        flow_gt, occ_gt = self.extract_flow_and_mask(flow_occ_fw)
        
        return img1, img2, flow_gt, occ_gt

    def read_and_decode_consistency(self, filename_queue):
        clean_img1_name = tf.string_join([self.img_dir, '/Clean/', filename_queue[0]])
        clean_img2_name = tf.string_join([self.img_dir, '/Clean/', filename_queue[1]])     
        clean_img1 = tf.image.decode_png(tf.read_file(clean_img1_name), channels=3)
        clean_img1 = tf.cast(clean_img1, tf.float32)
        clean_img2 = tf.image.decode_png(tf.read_file(clean_img2_name), channels=3)
        clean_img2 = tf.cast(clean_img2, tf.float32)    

        # rain_img1_name = tf.string_join([self.img_dir, '/HeavyRain_KITTI/Train/', filename_queue[0]])
        # rain_img2_name = tf.string_join([self.img_dir, '/HeavyRain_KITTI/Train/', filename_queue[1]])    

        rain_img1_name = tf.string_join([self.img_dir, '/LigntRain_KITTI/Train/', filename_queue[0]])
        rain_img2_name = tf.string_join([self.img_dir, '/LigntRain_KITTI/Train/', filename_queue[1]])     
        rain_img1 = tf.image.decode_png(tf.read_file(rain_img1_name), channels=3)
        rain_img1 = tf.cast(rain_img1, tf.float32)
        rain_img2 = tf.image.decode_png(tf.read_file(rain_img2_name), channels=3)
        rain_img2 = tf.cast(rain_img2, tf.float32)    

        derain_img1_name = tf.string_join([self.fake_flow_occ_dir, '/', filename_queue[0]])
        derain_img2_name = tf.string_join([self.fake_flow_occ_dir, '/', filename_queue[1]])     
        derain_img1 = tf.image.decode_png(tf.read_file(derain_img1_name), channels=3)
        derain_img1 = tf.cast(derain_img1, tf.float32)
        derain_img2 = tf.image.decode_png(tf.read_file(derain_img2_name), channels=3)
        derain_img2 = tf.cast(derain_img2, tf.float32)    
        
        return clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2  

    def augmentation(self, img1, img2):
        img1, img2 = random_crop([img1, img2], self.crop_h, self.crop_w)
        # img1.set_shape([256, 256, 3])
        # img2.set_shape([256, 256, 3])
        
        # img1, img2 = random_flip([img1, img2])
        # img1, img2 = random_channel_swap([img1, img2])
        return img1, img2 
    
    def augmentation_cycle(self, img1, img2):

        img1 = tf.image.resize_images(img1, size=(256, 256))
        img1.set_shape([256, 256, 3])
        
        img2 = tf.image.resize_images(img2, size=(256, 256))
        img2.set_shape([256, 256, 3])

        # 区别
        # images = [img1]
        # images = tf.train.shuffle_batch(
        #     [img1], batch_size=self.batch_size, num_threads=4,
        #     capacity=64,
        #     min_after_dequeue=1)
        return img1, img2
    
    def augmentation_distillation(self, img1, img2, flow_fw, flow_bw, occ_fw, occ_bw):
        [img1, img2, flow_fw, flow_bw, occ_fw, occ_bw] = random_crop([img1, img2, flow_fw, flow_bw, occ_fw, occ_bw], self.crop_h, self.crop_w)
        [img1, img2, occ_fw, occ_bw], [flow_fw, flow_bw] = random_flip_with_flow([img1, img2, occ_fw, occ_bw], [flow_fw, flow_bw])
        img1, img2 = random_channel_swap([img1, img2])
        return img1, img2, flow_fw, flow_bw, occ_fw, occ_bw
        
    def augmentation_consistency(self, clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2):
        [clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2] = random_crop([clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2], self.crop_h, self.crop_w)
        
        # img1, img2 = random_channel_swap([img1, img2])
        return clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2

    def augmentation_finetune(self, img1, img2, flow_gt, occ_gt):
        [img1, img2, flow_gt, occ_gt] = random_crop([img1, img2, flow_gt, occ_gt], self.crop_h, self.crop_w)
        # [img1, img2, occ_fw, occ_bw], [flow_fw, flow_bw] = random_flip_with_flow([img1, img2, occ_fw, occ_bw], [flow_fw, flow_bw])
        img1, img2 = random_channel_swap([img1, img2])
        return img1, img2, flow_gt, occ_gt

    def preprocess_augmentation(self, filename_queue):
        img1, img2 = self.read_and_decode(filename_queue)
        img1 = img1 / 255.       
        img2 = img2 / 255.  
        # 修改此处，修改为适配rain-kitti dataset， second stage第一次预训练
        img1, img2 = self.augmentation_cycle(img1, img2) 
        
        # img1, img2 = self.augmentation(img1, img2)
        return img1, img2

    def preprocess_augmentation_distillation(self, filename_queue):
        img1, img2, flow_fw, flow_bw, occ_fw, occ_bw = self.read_and_decode_distillation(filename_queue)
        img1 = img1 / 255.
        img2 = img2 / 255.        
        img1, img2, flow_fw, flow_bw, occ_fw, occ_bw = self.augmentation_distillation(img1, img2, flow_fw, flow_bw, occ_fw, occ_bw)
        return img1, img2, flow_fw, flow_bw, occ_fw, occ_bw  
    
    def preprocess_augmentation_consistency(self, filename_queue):
        clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2  = self.read_and_decode_consistency(filename_queue)
        clean_img1 = clean_img1 / 255.
        clean_img2 = clean_img2 / 255.       
        rain_img1 = rain_img1 / 255.
        rain_img2 = rain_img2 / 255.   
        derain_img1 = derain_img1 / 255.
        derain_img2 = derain_img2 / 255.    
        clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2 = self.augmentation_consistency(clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2)
        return clean_img1, clean_img2, rain_img1, rain_img2, derain_img1, derain_img2

    def preprocess_augmentation_finetune(self, filename_queue):
        img1, img2, flow_gt, occ_gt = self.read_and_decode_finetune(filename_queue)
        img1 = img1 / 255.
        img2 = img2 / 255.        
        img1, img2, flow_gt, occ_gt = self.augmentation_finetune(img1, img2, flow_gt, occ_gt)
        return img1, img2, flow_gt, occ_gt

    def preprocess_one_shot(self, filename_queue):
        img1, img2 = self.read_and_decode(filename_queue)
        img1 = img1 / 255.
        img2 = img2 / 255.  
              
        return img1, img2



    # cycleflow
    def create_batch_iterator(self, data_list, batch_size, shuffle=True, buffer_size=3000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator
    
    def create_batch_consistency_iterator(self, data_list, batch_size, shuffle=True, buffer_size=3000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation_consistency, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator
    
    def create_batch_distillation_iterator(self, data_list, batch_size, shuffle=True, buffer_size=5000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation_distillation, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator   

    def create_batch_finetune_iterator(self, data_list, batch_size, shuffle=True, buffer_size=3000, num_parallel_calls=4):
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_augmentation_finetune, num_parallel_calls=num_parallel_calls)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator    
    
    def create_one_shot_iterator(self, data_list, num_parallel_calls=4):
        #  For Validation or Testing
        #     Generate image and flow one_by_one without cropping, image and flow size may change every iteration
    
        data_list = tf.convert_to_tensor(data_list, dtype=tf.string)
        dataset = tf.data.Dataset.from_tensor_slices(data_list)
        dataset = dataset.map(self.preprocess_one_shot, num_parallel_calls=num_parallel_calls)        
        dataset = dataset.batch(1)
        dataset = dataset.repeat()
        iterator = dataset.make_initializable_iterator()
        return iterator    