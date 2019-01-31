# encoding=utf8
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
    model = load_model('model/nn4.small2.lrn.h5')

    lfw_pairs='pairs.txt'
    lfw_dir = 'data'
    lfw_file_ext='jpg'
    lfw_nrof_folds=2
    image_size=96
    batch_size=10

    # Read the file containing the pairs used for testing
    pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)

    embedding_size=128
    nrof_images = len(paths)
    print(actual_issame)

    nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
    print(nrof_batches)
    emb_array = np.zeros((nrof_images, embedding_size))

    for i in range(nrof_batches):
        start_index = i*batch_size
        end_index = min((i+1)*batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        #images = np.transpose(images, (0,3,1,2))
        t0 = time.time()
        y = model.predict_on_batch(images)
        emb_array[start_index:end_index,:] = y
        t1 = time.time()
        print('batch: ', i, ' time: ', t1-t0)
        print(len(emb_array))
    np.savetxt("emb_array.csv", emb_array, delimiter=",")

    emb_array = np.genfromtxt("C:\\Users\\ADMIN\\PycharmProjects\\FaceClassificationInMovie\\openface_keras\\emb_array.csv", delimiter=",")

    from sklearn import metrics
    from scipy.optimize import brentq
    from scipy import interpolate

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                                                         actual_issame,
                                                         nrof_folds=lfw_nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)
