import tensorflow as tf
from tensorflow.contrib import slim
from data_augmentation import flow_resize
from utils import lrelu
from warp import tf_warp
import layer
""" 
description
    building network
 """

# optical flow
def pyramid_processing(batch_img1, batch_img2, train=True, trainable=True, regularizer=None, is_scale=True):
    img_size = tf.shape(batch_img1)[1:3]
    x1_feature = layer.feature_extractor(batch_img1, train=train, trainable=trainable, regularizer=regularizer, name='feature_extractor')
    x2_feature = layer.feature_extractor(batch_img2, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    flow_estimated = layer._pyramid_processing(x1_feature, x2_feature, img_size, train=train, trainable=trainable, regularizer=regularizer, is_scale=is_scale)    
    return flow_estimated  

def pyramid_processing_bidirection(batch_img1, batch_img2, train=True, trainable=True, reuse=None, regularizer=None, is_scale=True):
    img_size = tf.shape(batch_img1)[1:3]
    x1_feature = layer.feature_extractor(batch_img1, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='feature_extractor')
    x2_feature = layer.feature_extractor(batch_img2, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='feature_extractor')
    
    flow_fw = layer._pyramid_processing(x1_feature, x2_feature, img_size, train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale)
    flow_bw = layer._pyramid_processing(x2_feature, x1_feature, img_size, train=train, trainable=trainable, reuse=True, regularizer=regularizer, is_scale=is_scale)
    return flow_fw, flow_bw
    

# rgb-rain-derain flow processing
def Rain_pyramid_processing_flow(batch_img1_rgb, batch_img2_rgb, batch_img1_rain, batch_img2_rain, batch_img1_derain, batch_img2_derain, train=True, trainable=True, reuse=None, regularizer=None, is_scale=True):
    img_rgb_size = tf.shape(batch_img1_rgb)[1:3]
    x1_rgb_feature = layer.feature_extractor(batch_img1_rgb, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='rgb_feature_extractor')
    x2_rgb_feature = layer.feature_extractor(batch_img2_rgb, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='rgb_feature_extractor')

    img_rain_size = tf.shape(batch_img1_rain)[1:3]
    x1_rain_feature = layer.feature_extractor(batch_img1_rain, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='rain_feature_extractor')
    x2_rain_feature = layer.feature_extractor(batch_img2_rain, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='rain_feature_extractor')

    img_derain_size = tf.shape(batch_img1_derain)[1:3]
    x1_derain_feature = layer.feature_extractor(batch_img1_derain, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='derain_feature_extractor')
    x2_derain_feature = layer.feature_extractor(batch_img2_derain, train=train, trainable=trainable, reuse=True, regularizer=regularizer, name='derain_feature_extractor')
    
    flow_fw = {}
    flow_bw = {}
    flow_fw['rgb'] = layer._pyramid_processing(x1_rgb_feature, x2_rgb_feature, img_rgb_size, train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale, name="rgb_pyramid_processing")
    flow_bw['rgb'] = layer._pyramid_processing(x2_rgb_feature, x1_rgb_feature, img_rgb_size, train=train, trainable=trainable, reuse=True, regularizer=regularizer, is_scale=is_scale, name="rgb_pyramid_processing")   

    flow_fw['rain'] = layer._pyramid_processing(x1_rain_feature, x2_rain_feature, img_rain_size, train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale, name="rain_pyramid_processing")
    flow_bw['rain'] = layer._pyramid_processing(x2_rain_feature, x1_rain_feature, img_rain_size, train=train, trainable=trainable, reuse=True, regularizer=regularizer, is_scale=is_scale, name="rain_pyramid_processing")    

    flow_fw['derain'] = layer._pyramid_processing(x1_derain_feature, x2_derain_feature, img_derain_size, train=train, trainable=trainable, reuse=None, regularizer=regularizer, is_scale=is_scale, name="derain_pyramid_processing")
    flow_bw['derain'] = layer._pyramid_processing(x2_derain_feature, x1_derain_feature, img_derain_size, train=train, trainable=trainable, reuse=True, regularizer=regularizer, is_scale=is_scale, name="derain_pyramid_processing")    


    return flow_fw, flow_bw



class OpticalFlow:
    def __init__(self, train=True, trainable=True, reuse=None, regularizer=None, is_scale=True, is_bidirection=False, name='flow'):
        self.train = train
        self.trainable = trainable
        self.reuse = reuse
        self.regularizer = regularizer
        self.is_scale = is_scale
        self.is_bidirection = is_bidirection
        self.name = name

    def __call__(self, input):
        with tf.variable_scope(self.name):
            batch_img1, batch_img2 = input
            if self.is_bidirection == False:
                img_size = tf.shape(batch_img1)[1:3]
                x1_feature = layer.feature_extractor(batch_img1, train=self.train, trainable=self.trainable, regularizer=self.regularizer, name='feature_extractor')
                x2_feature = layer.feature_extractor(batch_img2, train=self.train, trainable=self.trainable, reuse=True, regularizer=self.regularizer, name='feature_extractor')
                flow_estimated = layer._pyramid_processing(x1_feature, x2_feature, img_size, train=self.train, trainable=self.trainable, regularizer=self.regularizer, is_scale=self.is_scale, name='pyramid_processing') 
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
                return flow_estimated
            else:
                img_size = tf.shape(batch_img1)[1:3]
                x1_feature = layer.feature_extractor(batch_img1, train=self.train, trainable=self.trainable, reuse=self.reuse, regularizer=self.regularizer, name='feature_extractor')
                x2_feature = layer.feature_extractor(batch_img2, train=self.train, trainable=self.trainable, reuse=True, regularizer=self.regularizer, name='feature_extractor')
                
                flow_fw = layer._pyramid_processing(x1_feature, x2_feature, img_size, train=self.train, trainable=self.trainable, reuse=self.reuse, regularizer=self.regularizer, is_scale=self.is_scale, name='pyramid_processing')
                flow_bw = layer._pyramid_processing(x2_feature, x1_feature, img_size, train=self.train, trainable=self.trainable, reuse=True, regularizer=self.regularizer, is_scale=self.is_scale, name='pyramid_processing') 
                self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
                self.reuse = True
                return flow_fw, flow_bw


### cyclegan
class Generator:
    def __init__(self, name, is_training, ngf=64, norm='instance', image_size=128):
        self.name = name
        self.reuse = False
        self.ngf = ngf
        self.norm = norm
        self.is_training = is_training
        self.image_size = image_size

    def __call__(self, input, is_contra=False):
        """
        Args:
        input: batch_size x width x height x 3
        Returns:
        output: same size as input
        """
        with tf.variable_scope(self.name):
            # conv layers
            c7s1_32 = layer.c7s1_k(input, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='c7s1_32')                             # (?, w, h, 32)
            d64 = layer.dk(c7s1_32, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='d64')                                 # (?, w/2, h/2, 64)
            d128 = layer.dk(d64, 4*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='d128')                                # (?, w/4, h/4, 128)

            if self.image_size <= 128:
                # use 6 residual blocks for 128x128 images
                res_output = layer.n_res_blocks(d128, reuse=self.reuse, n=6)      # (?, w/4, h/4, 128)
            else:
                # 9 blocks for higher resolution
                res_output = layer.n_res_blocks(d128, reuse=self.reuse, n=9)      # (?, w/4, h/4, 128)

            # fractional-strided convolution
            u64 = layer.uk(res_output, 2*self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u64')                                 # (?, w/2, h/2, 64)
            u32 = layer.uk(u64, self.ngf, is_training=self.is_training, norm=self.norm,
                reuse=self.reuse, name='u32', output_size=self.image_size)         # (?, w, h, 32)

            # conv layer
            # Note: the paper said that ReLU and _norm were used
            # but actually tanh was used and no _norm here
            output = layer.c7s1_k(u32, 3, norm=None,
                activation='tanh', reuse=self.reuse, name='output')           # (?, w, h, 3)
        # set reuse=True for next call
        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        
        if is_contra == False:
            return output
        elif is_contra == True:
            return res_output

    def sample(self, input):
        image = utils.batch_convert2int(self.__call__(input))
        image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
        return image
    

class Discriminator:
    def __init__(self, name, is_training, norm='instance', use_sigmoid=False):
        self.name = name
        self.is_training = is_training
        self.norm = norm
        self.reuse = False
        self.use_sigmoid = use_sigmoid

    def __call__(self, input):
        """
        Args:
        input: batch_size x image_size x image_size x 3
        Returns:
        output: 4D tensor batch_size x out_size x out_size x 1 (default 1x5x5x1)
                filled with 0.9 if real, 0.0 if fake
        """
        with tf.variable_scope(self.name):
            # convolution layers
            C64 = layer.Ck(input, 64, reuse=self.reuse, norm=None,
                is_training=self.is_training, name='C64')             # (?, w/2, h/2, 64)
            C128 = layer.Ck(C64, 128, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='C128')            # (?, w/4, h/4, 128)
            C256 = layer.Ck(C128, 256, reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='C256')            # (?, w/8, h/8, 256)
            C512 = layer.Ck(C256, 512,reuse=self.reuse, norm=self.norm,
                is_training=self.is_training, name='C512')            # (?, w/16, h/16, 512)
        
            # apply a convolution to produce a 1 dimensional output (1 channel?)
            # use_sigmoid = False if use_lsgan = True
            output = layer.last_conv(C512, reuse=self.reuse,
                use_sigmoid=self.use_sigmoid, name='output')          # (?, w/16, h/16, 1)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output


class Contra_Network:
    def __init__(self, name, num_channels, is_training, generator):
        self.name = name
        self.num_channels = num_channels
        self.is_training = is_training
        self.generator = generator
        self.reuse = False

    def __call__(self, input):
        """
        Args:
        input: batch_size x image_size/4 x image_size/4 x channels
        Returns:
        output: 4D tensor batch_size x num_channels
                filled with 0.9 if real, 0.0 if fake
        """
        # 实例化生成器最后残差特征层输出
        contra_feature = self.generator(input, is_contra=True)
        with tf.variable_scope(self.name):
            # MLP layers
            contra_feature = slim.max_pool2d(contra_feature, [2, 2], stride=2, padding='SAME',scope='pool_1')
            output = layer.fully_connect(contra_feature, self.num_channels, reuse=self.reuse, is_training=self.is_training, name='F256') # (?, 256)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output

class Patch_Contra_Network:
    def __init__(self, name, num_channels, is_training, generator):
        self.name = name
        self.num_channels = num_channels
        self.is_training = is_training
        self.generator = generator
        self.reuse = False

    def __call__(self, input):
        """
        Args:
        input: batch_size x image_size/4 x image_size/4 x channels
        Returns:
        output: 4D tensor batch_size x num_channels
                filled with 0.9 if real, 0.0 if fake
        """
        # 实例化生成器最后残差特征层输出
        contra_feature = self.generator(input, is_contra=True)
        with tf.variable_scope(self.name):
            # MLP layers
            contra_feature = slim.max_pool2d(contra_feature, [2, 2], stride=2, padding='SAME',scope='pool_1')
            output = layer.fully_connect(contra_feature, self.num_channels, reuse=self.reuse, is_training=self.is_training, name='F256') # (?, 256)

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
# tf.layers.dense()