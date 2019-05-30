import sys
sys.path.insert(0, './lib/')

from extract_patches import extract_ordered_overlap, recompone_overlap
from help_functions import *

import numpy as np
from PIL import Image

img = np.asarray(Image.open('./DRIVE/test/images/01.png').convert('L'))
img = np.reshape(img, (1, 1, img.shape[0], img.shape[1])) / 255.
img_data, new_image_size, n_subimgs = extract_ordered_overlap(img, (48, 48), (5, 5))

print(img_data.shape)
print(new_image_size)
print(n_subimgs)

pred_imgs = recompone_overlap(
    img_data,
    new_image_size[0],
    new_image_size[1],
    5,
    5
)

visualize(group_images(pred_imgs, 1), "_all_predictions").show()
