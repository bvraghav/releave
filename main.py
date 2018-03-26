## Usage:
##   main IMAGE MASK_1 MASK_2 [NEIGHBOURHOOD_SIZE=5]

import sys
import logging as lg
lg.basicConfig(level=lg.INFO, format='%(levelname)-8s: %(message)s')

## Args
if len(sys.argv) < 4 :
  lg.fatal('Usage:\n  main IMAGE MASK_1 MASK_2 [NEIGHBOURHOOD_SIZE=5 N_SAMPLES=10]')
  sys.exit(64)

from image_functions import open_image, bb_from_point, kernel_create
from lum import diff_lum
import numpy as np
import random

def random_pairs(l1, l2, count):
  indices = np.ndarray([count, 2], dtype=np.int64)

  for i in range(count) :
    indices[i][0] = random.randrange(l1)
    indices[i][1] = random.randrange(l2)

  return indices

image  = open_image(sys.argv[1])
mask_1 = open_image(sys.argv[2])
mask_2 = open_image(sys.argv[3])
k_size = int(sys.argv[4])     \
         if len(sys.argv) > 4 \
            else 5
n_samples = int(sys.argv[5])     \
         if len(sys.argv) > 5 \
            else 10

lg.info('Using params: k_size: %d, n_samples: %d', k_size, n_samples)

if (image is None or
    mask_1 is None or
    mask_2 is None):
  lg.fatal('Error opening one of the images. Exiting')
  sys.exit(65)

points_1 = np.nonzero(mask_1)
points_1 = np.array(points_1, dtype=np.int32).T

points_2 = np.nonzero(mask_2)
points_2 = np.array(points_2, dtype=np.int32).T

indices = random_pairs(points_1.shape[0], points_2.shape[0], n_samples)

kernel = kernel_create(k_size)
image_extents = image.shape[0:2]

lum_diff = np.ndarray([n_samples], dtype=np.float32)
for i in range(n_samples) :
  p_m1 = points_1[indices[i][0]]
  p_m2 = points_2[indices[i][1]]

  p0, p1 = bb_from_point(p_m1, k_size, image_extents)
  y0, x0, y1, x1 = list(p0) + list(p1)
  part_mask_1 = mask_1[y0:y1, x0:x1]
  part_image_1 = image[y0:y1, x0:x1]

  p0, p1 = bb_from_point(p_m2, k_size, image_extents)
  y0, x0, y1, x1 = list(p0) + list(p1)
  part_mask_2 = mask_2[y0:y1, x0:x1]
  part_image_2 = image[y0:y1, x0:x1]

  lum_diff[i] = diff_lum(part_image_1, part_mask_1,
                         part_image_2, part_mask_2,
                         kernel)

lg.info('Luminosity Difference (min, max, mu, sigma): %0.5g, %0.5g, %0.5g, %0.5g',
        np.min(lum_diff), np.max(lum_diff),
        np.average(lum_diff), np.std(lum_diff))
