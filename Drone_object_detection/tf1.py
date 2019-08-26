# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 14:59:58 2019

@author: Durgesh
"""

import os
import sys
import time
import random

import tensorflow as tf
import numpy as np
import time
from PIL import Image, ImageDraw
from object_detection.utils import label_map_util, visualization_utils
import glob
from six.moves import urllib
import tarfile

##### Constants
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
MODEL_NAME = "ssd_mobilenet_v1_coco_2017_11_17"
PATH_TO_OBJECT_DETECTION_REPO = ""  # Insert path to tensorflow object detection repository - models/research/object_detection/ 
PATH_TO_LABELS = PATH_TO_OBJECT_DETECTION_REPO + "data/mscoco_label_map.pbtxt"
NUM_CLASSES = 1

##### config variables to set 
threshold = 0.5
test_image_dir = ""    # Insert path to directory containing test images

def download_model(model_name):
    """Download the model from tensorflow model zoo
    Args:
        model_name: name of model to download
    """
    model_file = model_name + '.tar.gz'
    if os.path.isfile(model_name + '/frozen_inference_graph.pb'):
        print("File already downloaded")
        return
    opener = urllib.request.URLopener()
    try:
        print("Downloading Model")
        opener.retrieve(DOWNLOAD_BASE + model_file, model_file)
        print("Extracting Model")
        tar_file = tarfile.open(model_file)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())
        print("Done")
    except:
        raise Exception("Not able to download model, please check the model name")