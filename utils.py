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

def imshow(img, re_normalize=False):
    if re_normalize:
        min_value = np.min(img)
        max_value = np.max(img)
        img = (img - min_value) / (max_value - min_value)
        img = img * 255
    elif np.max(img) <= 1.:
        img = img * 255
    img = img.astype('uint8')
    shape = img.shape
    if len(shape) == 2:
        img = np.repeat(np.expand_dims(img, -1), 3, -1)
    elif shape[2] == 1:
        img = np.repeat(img, 3, -1)
    plt.imshow(img)
    plt.show()
    
def rgb_bgr(img):
    tmp = np.copy(img[:, :, 0])
    img[:, :, 0] = np.copy(img[:, :, 2])
    img[:, :, 2] = np.copy(tmp)  
    return img
  
def compute_Fl(flow_gt, flow_est, mask):
    # F1 measure
    err = tf.multiply(flow_gt - flow_est, mask)
    err_norm = tf.norm(err, axis=-1)
    
    flow_gt_norm = tf.maximum(tf.norm(flow_gt, axis=-1), 1e-12)
    F1_logic = tf.logical_and(err_norm > 3, tf.divide(err_norm, flow_gt_norm) > 0.05)
    F1_logic = tf.cast(tf.logical_and(tf.expand_dims(F1_logic, -1), mask > 0), tf.float32)
    F1 = tf.reduce_sum(F1_logic) / (tf.reduce_sum(mask) + 1e-6)
    return F1    

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)
        if grads != []:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def length_sq(x):
    return tf.reduce_sum(tf.square(x), 3, keepdims=True)    

def occlusion(flow_fw, flow_bw):
    x_shape = tf.shape(flow_fw)
    H = x_shape[1]
    W = x_shape[2]    
    flow_bw_warped = tf_warp(flow_bw, flow_fw, H, W)
    flow_fw_warped = tf_warp(flow_fw, flow_bw, H, W)
    flow_diff_fw = flow_fw + flow_bw_warped
    flow_diff_bw = flow_bw + flow_fw_warped
    mag_sq_fw = length_sq(flow_fw) + length_sq(flow_bw_warped)
    mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    occ_thresh_fw =  0.01 * mag_sq_fw + 0.5
    occ_thresh_bw =  0.01 * mag_sq_bw + 0.5
    occ_fw = tf.cast(length_sq(flow_diff_fw) > occ_thresh_fw, tf.float32)
    occ_bw = tf.cast(length_sq(flow_diff_bw) > occ_thresh_bw, tf.float32)

    return occ_fw, occ_bw

def flow_warp_error(img1, img2, flow_fw):
    x_shape = tf.shape(img1)
    H = x_shape[1]
    W = x_shape[2]    
    img2_warped = tf_warp(img2, flow_fw, H, W)
    
    image_diff = img1 - img2_warped
    # image_diff = img2_warped
    
    # mag_sq_fw = length_sq(img1) + length_sq(img2_warped)
    # mag_sq_bw = length_sq(flow_bw) + length_sq(flow_fw_warped)
    # thresh_fw =  0.01 * mag_sq_fw + 0.5
    # warp_error = tf.cast(length_sq(image_diff) > thresh_fw, tf.float32)
    warp_error = image_diff

    return warp_error

def rgb_bgr(img):
    tmp = np.copy(img[:, :, 0])
    img[:, :, 0] = np.copy(img[:, :, 2])
    img[:, :, 2] = np.copy(tmp)  
    return img

# evalution metric
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

def epe_metric(gt_flow, pred_flow):
    def flow_error(tu, tv, u, v):
        """
        Calculate average end point error
        :param tu: ground-truth horizontal flow map
        :param tv: ground-truth vertical flow map
        :param u:  estimated horizontal flow map
        :param v:  estimated vertical flow map
        :return: End point error of the estimated flow
        """
        smallflow = 0.0
        '''
        stu = tu[bord+1:end-bord,bord+1:end-bord]
        stv = tv[bord+1:end-bord,bord+1:end-bord]
        su = u[bord+1:end-bord,bord+1:end-bord]
        sv = v[bord+1:end-bord,bord+1:end-bord]
        '''
        stu = tu[:]
        stv = tv[:]
        su = u[:]
        sv = v[:]

        idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (abs(stv) > UNKNOWN_FLOW_THRESH)
        stu[idxUnknow] = 0
        stv[idxUnknow] = 0
        su[idxUnknow] = 0
        sv[idxUnknow] = 0

        ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
        index_su = su[tuple(ind2)]
        index_sv = sv[tuple(ind2)]
        an = 1.0 / np.sqrt(index_su ** 2 + index_sv ** 2 + 1)
        un = index_su * an
        vn = index_sv * an

        index_stu = stu[tuple(ind2)]
        index_stv = stv[tuple(ind2)]
        tn = 1.0 / np.sqrt(index_stu ** 2 + index_stv ** 2 + 1)
        tun = index_stu * tn
        tvn = index_stv * tn

        '''
        angle = un * tun + vn * tvn + (an * tn)
        index = [angle == 1.0]
        angle[index] = 0.999
        ang = np.arccos(angle)
        mang = np.mean(ang)
        mang = mang * 180 / np.pi
        '''

        epe = np.sqrt((stu - su) ** 2 + (stv - sv) ** 2)
        epe = epe[tuple(ind2)]
        mepe = np.mean(epe)
        return mepe
    
    average_epe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1], pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_epe


def error_pix_metric(predict_flow, flow_gt, max_pixel_threshold):
    diff = np.absolute(predict_flow - flow_gt)
    mask = np.sum(diff, axis=2)
    
    num = np.sum(mask > max_pixel_threshold)
    num_percent = num / (flow_gt.shape[0]*flow_gt.shape[1])

    return num_percent



# related to cyclegan
def convert2int(image):
  """ Transfrom from float tensor ([-1.,1.]) to int image ([0,255])
  """
  return tf.image.convert_image_dtype((image+1.0)/2.0, tf.uint8)

def convert2float(image):
  """ Transfrom from int image ([0,255]) to float tensor ([-1.,1.])
  """
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  return (image/127.5) - 1.0

def batch_convert2int(images):
  """
  Args:
    images: 4D float tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D int tensor
  """
  return tf.map_fn(convert2int, images, dtype=tf.uint8)

def batch_convert2float(images):
  """
  Args:
    images: 4D int tensor (batch_size, image_size, image_size, depth)
  Returns:
    4D float tensor
  """
  return tf.map_fn(convert2float, images, dtype=tf.float32)

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image

