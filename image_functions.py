import logging as lg

import cv2
import numpy as np

def open_image(image_name) :
  img = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
  if img is None :
    lg.warn('Problem opening: %s', image_name)
    raise Exception('Problem opening: %s' % image_name)

  lg.info('Success opening : %s, %s', image_name, img.shape)

  img = img.astype(np.float32) / 255
  lg.info('@`%s\' (min, max, mu, sigma): %0.2g, %0.2g, %0.2g, %0.2g',
          image_name, np.min(img), np.max(img),
          np.average(img), np.std(img))

  return img

def kernel_create(k_size) :
  K = cv2.getGaussianKernel(k_size, 0)
  K = K * K.T
  lg.debug('Gaussian Kernel: size: %s, sum: %0.2g', K.shape, np.sum(K))

  return K


def kernel_from_image(image):
  l_d0 = image.shape[0]
  return kernel_create(l_d0)

def bb_from_point(point, bb_size, image_extents) :
  pt = np.array(point) - (bb_size // 2)
  pt = np.where(pt < 0, [0, 0], pt)
  image_extents = np.array(image_extents)
  pt = np.where(pt + bb_size > image_extents, image_extents - bb_size, pt)
  return pt, pt + bb_size
  
