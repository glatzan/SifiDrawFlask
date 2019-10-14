#!/usr/bin/python

import numpy as np
from PIL import Image
from skimage.io import imsave
from skimage import img_as_uint

from tf_unet import unet

net = unet.Unet(channels=1, n_class=2, layers=3, features_root=24)

img = np.array(Image.open("test/194.png"), np.float32)
ny = img.shape[0]
nx = img.shape[1]
img = img.reshape(1, ny, nx, 1)
img -= np.amin(img)
img /= np.amax(img)

prediction = net.predict("model/unet_graft/model.ckpt", img)

mask = prediction[0, ..., 1] > 0.1
imsave("194_unet.png", img_as_uint(mask))

