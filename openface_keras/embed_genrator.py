import os
import numpy as np
import math
from openface_keras import facenet
from openface_keras import lfw


import time
import tensorflow as tf
from keras.models import load_model
from keras.utils import CustomObjectScope

with CustomObjectScope({'tf': tf}):
    model = load_model('./model/nn4.small2.v1.h5')

    lfw_pairs='data/pairs.txt'
    lfw_dir='data/dlib-affine-sz'
    lfw_file_ext='png'
    lfw_nrof_folds=10
    image_size=96
    batch_size=100

    # Read the file containing the pairs used for testing
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)

    embedding_size=128
    nrof_images = len(paths)
    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    emb_array = np.zeros((nrof_images, embedding_size))

    for i in range(nrof_batches):
      start_index = i*batch_size
      end_index = min((i+1)*batch_size, nrof_images)
      paths_batch = paths[start_index:end_index]
      images = facenet.load_data(paths_batch, False, False, image_size)
      images = np.transpose(images, (0,3,1,2))

      t0 = time.time()
      y = model.predict_on_batch(images)
      emb_array[start_index:end_index,:] = y
      t1 = time.time()

      print('batch: ', i, ' time: ', t1-t0)