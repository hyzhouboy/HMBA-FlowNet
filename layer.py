import tensorflow as tf 
from tensorflow.contrib import slim
from data_augmentation import flow_resize
from utils import lrelu
from warp import tf_warp
from helper import _weights, _biases, _leaky_relu, _norm, _batch_norm, _instance_norm, safe_log


### optical flow
def feature_extractor(x, train=True, trainable=True, reuse=None, regularizer=None, name='feature_extractor'):
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):
            net = {}
            net['conv1_1'] = slim.conv2d(x, 16, stride=2, scope='conv1_1')
            net['conv1_2'] = slim.conv2d(net['conv1_1'], 16, stride=1, scope='conv1_2')
            
            net['conv2_1'] = slim.conv2d(net['conv1_2'], 32, stride=2, scope='conv2_1')
            net['conv2_2'] = slim.conv2d(net['conv2_1'], 32, stride=1, scope='conv2_2')
            
            net['conv3_1'] = slim.conv2d(net['conv2_2'], 64, stride=2, scope='conv3_1')
            net['conv3_2'] = slim.conv2d(net['conv3_1'], 64, stride=1, scope='conv3_2')                

            net['conv4_1'] = slim.conv2d(net['conv3_2'], 96, stride=2, scope='conv4_1')
            net['conv4_2'] = slim.conv2d(net['conv4_1'], 96, stride=1, scope='conv4_2')                  
            
            net['conv5_1'] = slim.conv2d(net['conv4_2'], 128, stride=2, scope='conv5_1')
            net['conv5_2'] = slim.conv2d(net['conv5_1'], 128, stride=1, scope='conv5_2') 
            
            net['conv6_1'] = slim.conv2d(net['conv5_2'], 192, stride=2, scope='conv6_1')
            net['conv6_2'] = slim.conv2d(net['conv6_1'], 192, stride=1, scope='conv6_2')  
    
    return net

def context_network(x, flow, train=True, trainable=True, reuse=None, regularizer=None, name='context_network'):
    x_input = tf.concat([x, flow], axis=-1)
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):        
            net = {}
            net['dilated_conv1'] = slim.conv2d(x_input, 128, rate=1, scope='dilated_conv1')
            net['dilated_conv2'] = slim.conv2d(net['dilated_conv1'], 128, rate=2, scope='dilated_conv2')
            net['dilated_conv3'] = slim.conv2d(net['dilated_conv2'], 128, rate=4, scope='dilated_conv3')
            net['dilated_conv4'] = slim.conv2d(net['dilated_conv3'], 96, rate=8, scope='dilated_conv4')
            net['dilated_conv5'] = slim.conv2d(net['dilated_conv4'], 64, rate=16, scope='dilated_conv5')
            net['dilated_conv6'] = slim.conv2d(net['dilated_conv5'], 32, rate=1, scope='dilated_conv6')
            net['dilated_conv7'] = slim.conv2d(net['dilated_conv6'], 2, rate=1, activation_fn=None, scope='dilated_conv7')
    
    refined_flow = net['dilated_conv7'] + flow
    
    return refined_flow


def get_shape(x, train=True):
    if train:
        x_shape = x.get_shape().as_list()
    else:
        x_shape = tf.shape(x)      
    return x_shape
    

def estimator(x1, x2, flow, train=True, trainable=True, reuse=None, regularizer=None, name='estimator'):
    # warp x2 according to flow
    x_shape = get_shape(x1, train=train)
    H = x_shape[1]
    W = x_shape[2]
    channel = x_shape[3]
    x2_warp = tf_warp(x2, flow, H, W)
    
    # ---------------cost volume-----------------
    # normalize
    x1 = tf.nn.l2_normalize(x1, axis=3)
    x2_warp = tf.nn.l2_normalize(x2_warp, axis=3)        
    d = 9
    
    # choice 1: use tf.extract_image_patches, may not work for some tensorflow versions
    x2_patches = tf.extract_image_patches(x2_warp, [1, d, d, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1], padding='SAME')
    
    # choice 2: use convolution, but is slower than choice 1
    # out_channels = d * d
    # w = tf.eye(out_channels*channel, dtype=tf.float32)
    # w = tf.reshape(w, (d, d, channel, out_channels*channel))
    # x2_patches = tf.nn.conv2d(x2_warp, w, strides=[1, 1, 1, 1], padding='SAME')
    
    x2_patches = tf.reshape(x2_patches, [-1, H, W, d, d, channel])
    x1_reshape = tf.reshape(x1, [-1, H, W, 1, 1, channel])
    x1_dot_x2 = tf.multiply(x1_reshape, x2_patches)
    cost_volume = tf.reduce_sum(x1_dot_x2, axis=-1)
    cost_volume = tf.reshape(cost_volume, [-1, H, W, d*d])
    
    # --------------estimator network-------------
    net_input = tf.concat([cost_volume, x1, flow], axis=-1)
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):        
            net = {}
            net['conv1'] = slim.conv2d(net_input, 128, scope='conv1')
            net['conv2'] = slim.conv2d(net['conv1'], 128, scope='conv2')
            net['conv3'] = slim.conv2d(net['conv2'], 96, scope='conv3')
            net['conv4'] = slim.conv2d(net['conv3'], 64, scope='conv4')
            net['conv5'] = slim.conv2d(net['conv4'], 32, scope='conv5')
            net['conv6'] = slim.conv2d(net['conv5'], 2, activation_fn=None, scope='conv6')
    
    #flow_estimated = net['conv6']
    
    return net

# residual network, contain six block, the shape of input is equal to the shape of output
def residual_network(x, train=True, trainable=True, reuse=None, regularizer=None, name='residual_network'):
    x_shape = x.get_shape().as_list()
    channel = x_shape[3]
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, kernel_size=3, padding='SAME', trainable=trainable):        
            net = {}
            # block 1
            net['residual_block1_1'] = slim.conv2d(x, channel, stride=1, scope='residual_block1_1')
            net['residual_block1_2'] = slim.conv2d(net['residual_block1_1'], channel, stride=1, activation_fn=None, scope='residual_block1_2') + x
            net['residual_block1_relu'] = lrelu(net['residual_block1_2'])
            # block 2
            net['residual_block2_1'] = slim.conv2d(net['residual_block1_relu'], channel, stride=1, scope='residual_block2_1')
            net['residual_block2_2'] = slim.conv2d(net['residual_block2_1'], channel, stride=1, activation_fn=None, scope='residual_block2_2') + net['residual_block1_relu']
            net['residual_block2_relu'] = lrelu(net['residual_block2_2'])
            # block 3
            net['residual_block3_1'] = slim.conv2d(net['residual_block2_relu'], channel, stride=1, scope='residual_block3_1')
            net['residual_block3_2'] = slim.conv2d(net['residual_block3_1'], channel, stride=1, activation_fn=None, scope='residual_block3_2') + net['residual_block2_relu']
            net['residual_block3_relu'] = lrelu(net['residual_block3_2'])
            # block 4
            net['residual_block4_1'] = slim.conv2d(net['residual_block3_relu'], channel, stride=1, scope='residual_block4_1')
            net['residual_block4_2'] = slim.conv2d(net['residual_block4_1'], channel, stride=1, activation_fn=None, scope='residual_block4_2') + net['residual_block3_relu']
            net['residual_block4_relu'] = lrelu(net['residual_block4_2'])
            # block 5
            net['residual_block5_1'] = slim.conv2d(net['residual_block4_relu'], channel, stride=1, scope='residual_block5_1')
            net['residual_block5_2'] = slim.conv2d(net['residual_block5_1'], channel, stride=1, activation_fn=None, scope='residual_block5_2') + net['residual_block4_relu']
            net['residual_block5_relu'] = lrelu(net['residual_block5_2'])
            # block 6
            net['residual_block6_1'] = slim.conv2d(net['residual_block5_relu'], channel, stride=1, scope='residual_block6_1')
            net['residual_block6_2'] = slim.conv2d(net['residual_block6_1'], channel, stride=1, activation_fn=None, scope='residual_block6_2') + net['residual_block5_relu']
            net['residual_block6_relu'] = lrelu(net['residual_block6_2'])   
    out_features = net['residual_block6_relu']
    return out_features

# the discriminator of transformation module
""" def discriminator(x, train=True, trainable=True, reuse=None, regularizer=None, name='discriminator_network'):
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        with slim.arg_scope([slim.conv2d], activation_fn=lrelu, normalizer_fn=slim.batch_norm, kernel_size=3, padding='SAME', trainable=trainable):        
            net = {}
            # discriminator
            net['discriminator_conv1'] = slim.conv2d(x, 64, stride=2, scope='discriminator_conv1')
            net['discriminator_conv2'] = slim.conv2d(net['discriminator_conv1'], 128, stride=2, scope='discriminator_conv2')
            net['discriminator_conv3'] = slim.conv2d(net['discriminator_conv2'], 256, stride=2, scope='discriminator_conv3')
            # net['discriminator_conv4'] = slim.conv2d(net['discriminator_conv3'], 512, stride=2, scope='discriminator_conv4')
            net['discriminator_conv4'] = slim.conv2d(net['discriminator_conv3'], 1, stride=1, normalizer_fn=None, scope='discriminator_conv4')
            
    dis_result = net['discriminator_conv4']
    # dis_sigmoid = tf.nn.sigmoid(dis_result)
    return dis_result 
"""


def _pyramid_processing(x1_feature, x2_feature, img_size, train=True, trainable=True, reuse=None, regularizer=None, is_scale=True, name='pyramid_processing'):
    with tf.variable_scope(name, reuse=reuse, regularizer=regularizer):
        x_shape = tf.shape(x1_feature['conv6_2'])

        initial_flow = tf.zeros([x_shape[0], x_shape[1], x_shape[2], 2], dtype=tf.float32, name='initial_flow')
        flow_estimated = {}
        flow_estimated['level_6'] = estimator(x1_feature['conv6_2'], x2_feature['conv6_2'], 
            initial_flow, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='estimator_level_6')['conv6']
        
        for i in range(4):
            feature_name = 'conv%d_2' % (5-i)
            feature_size = tf.shape(x1_feature[feature_name])[1:3]
            initial_flow = flow_resize(flow_estimated['level_%d' % (6-i)], feature_size, is_scale=is_scale)
            if i == 3:
                estimator_net_level_2 = estimator(x1_feature[feature_name], x2_feature[feature_name], 
                    initial_flow, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='estimator_level_%d' % (5-i))
                flow_estimated['level_2'] = estimator_net_level_2['conv6']
            else:
                flow_estimated['level_%d' % (5-i)] = estimator(x1_feature[feature_name], x2_feature[feature_name], 
                    initial_flow, train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='estimator_level_%d' % (5-i))['conv6']
        
        x_feature = estimator_net_level_2['conv5']
        flow_estimated['refined'] = context_network(x_feature, flow_estimated['level_2'], train=train, trainable=trainable, reuse=reuse, regularizer=regularizer, name='context_network')
        flow_estimated['full_res'] = flow_resize(flow_estimated['refined'], img_size, is_scale=is_scale)     
        
    return flow_estimated


# cyclegan module
### Generator layers
def c7s1_k(input, k, reuse=False, norm='instance', activation='relu', is_training=True, name='c7s1_k'):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[7, 7, input.get_shape()[3], k])

    padded = tf.pad(input, [[0,0],[3,3],[3,3],[0,0]], 'REFLECT')
    conv = tf.nn.conv2d(padded, weights,
        strides=[1, 1, 1, 1], padding='VALID')

    normalized = _norm(conv, is_training, norm)

    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    return output

def dk(input, k, reuse=False, norm='instance', is_training=True, name=None):
  """ A 3x3 Convolution-BatchNorm-ReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
    name: string, e.g. 'd64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 2, 2, 1], padding='SAME')
    normalized = _norm(conv, is_training, norm)
    output = tf.nn.relu(normalized)
    return output

def Rk(input, k,  reuse=False, norm='instance', is_training=True, name=None):
  """ A residual block that contains two 3x3 convolutional layers
      with the same number of filters on both layer
  Args:
    input: 4D Tensor
    k: integer, number of filters (output depth)
    reuse: boolean
    name: string
  Returns:
    4D tensor (same shape as input)
  """
  with tf.variable_scope(name, reuse=reuse):
    with tf.variable_scope('layer1', reuse=reuse):
      weights1 = _weights("weights1",
        shape=[3, 3, input.get_shape()[3], k])
      padded1 = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv1 = tf.nn.conv2d(padded1, weights1,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized1 = _norm(conv1, is_training, norm)
      relu1 = tf.nn.relu(normalized1)

    with tf.variable_scope('layer2', reuse=reuse):
      weights2 = _weights("weights2",
        shape=[3, 3, relu1.get_shape()[3], k])

      padded2 = tf.pad(relu1, [[0,0],[1,1],[1,1],[0,0]], 'REFLECT')
      conv2 = tf.nn.conv2d(padded2, weights2,
          strides=[1, 1, 1, 1], padding='VALID')
      normalized2 = _norm(conv2, is_training, norm)
    output = input+normalized2
    return output

def n_res_blocks(input, reuse, norm='instance', is_training=True, n=6):
  depth = input.get_shape()[3]
  for i in range(1,n+1):
    output = Rk(input, depth, reuse, norm, is_training, 'R{}_{}'.format(depth, i))
    input = output
  return output

from tensorflow.contrib import slim

def uk(input, k, reuse=False, norm='instance', is_training=True, name=None, output_size=None):
  """ A 3x3 fractional-strided-Convolution-BatchNorm-ReLU layer
      with k filters, stride 1/2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    with slim.arg_scope([slim.conv2d_transpose], activation_fn=None, kernel_size=3, padding='SAME', trainable=is_training):
      fsconv = slim.conv2d_transpose(input, k, stride=2)

      # source code, edit new version for dataset
      """ input_shape = input.get_shape().as_list()
      weights = _weights("weights",
        shape=[3, 3, k, input_shape[3]])

      if not output_size:
        output_size = input_shape[1]*2
      output_shape = [input_shape[0], output_size, output_size, k]
      fsconv = tf.nn.conv2d_transpose(input, weights,
          output_shape=output_shape,
          strides=[1, 2, 2, 1], padding='SAME') 
      """
      normalized = _norm(fsconv, is_training, norm)
      output = tf.nn.relu(normalized)
    return output

### Discriminator layers
def Ck(input, k, slope=0.2, stride=2, reuse=False, norm='instance', is_training=True, name=None):
  """ A 4x4 Convolution-BatchNorm-LeakyReLU layer with k filters and stride 2
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    slope: LeakyReLU's slope
    stride: integer
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'C64'
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], k])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, stride, stride, 1], padding='SAME')

    normalized = _norm(conv, is_training, norm)
    output = _leaky_relu(normalized, slope)
    return output

def last_conv(input, reuse=False, use_sigmoid=False, name=None):
  """ Last convolutional layer of discriminator network
      (1 filter with size 4x4, stride 1)
  Args:
    input: 4D tensor
    reuse: boolean
    use_sigmoid: boolean (False if use lsgan)
    name: string, e.g. 'C64'
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[4, 4, input.get_shape()[3], 1])
    biases = _biases("biases", [1])

    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='SAME')
    output = conv + biases
    if use_sigmoid:
      output = tf.sigmoid(output)
    return output

# contra network layer
def fully_connect(input, k, reuse=False, is_training=True, name=None):
  """ A fully connect layer
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    is_training: boolean or BoolTensor
    reuse: boolean
    name: string, e.g. 'c7sk-32'
    output_size: integer, desired output size of layer
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    with slim.arg_scope([slim.fully_connected], trainable=is_training):
      h_feat = slim.flatten(input)

      # 这里开始,写2层全连接层
      digits = slim.fully_connected(h_feat, k, activation_fn=tf.nn.relu)
      digits = slim.fully_connected(digits, k, activation_fn=None)
      # source code, edit new version for dataset
      """ input_shape = input.get_shape().as_list()
      weights = _weights("weights",
        shape=[3, 3, k, input_shape[3]])

      if not output_size:
        output_size = input_shape[1]*2
      output_shape = [input_shape[0], output_size, output_size, k]
      fsconv = tf.nn.conv2d_transpose(input, weights,
          output_shape=output_shape,
          strides=[1, 2, 2, 1], padding='SAME') 
      """
      output = tf.nn.l2_normalize(digits, dim=1)
      # normalized = _norm(fsconv, is_training, norm)
      # output = tf.nn.relu(normalized)
    return output