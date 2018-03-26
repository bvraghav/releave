import logging as lg
import numpy as np

def reduce_lum(obj, mask, kernel) :
  kernel_1 = kernel * mask
  kernel_1 /= np.sum(kernel_1)
  lg.debug('Sum of Gaussian Kernel: %0.2g', np.sum(kernel_1))

  luminosity_mat = kernel_1 * obj
  luminosity = np.sum(luminosity_mat)
  lg.debug ('Luminosity Score of obj at center: %0.2g', luminosity)

  return luminosity


def diff_lum(obj_1, mask_1, obj_2, mask_2, kernel) :

  lum_1 = reduce_lum(obj_1, mask_1, kernel)
  lum_2 = reduce_lum(obj_2, mask_2, kernel)

  return abs(lum_1 - lum_2)

