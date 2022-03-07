import os
import shutil
import pathlib
import argparse
import configparser

import math
import time
import tqdm
import logging
import collections
from pprint import pprint
from IPython.display import display

import re
import string

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as tf_hub
import tensorflow_text as tf_text
import tensorflow_datasets as tfds

### Setup

# suppress warnings 

logging.getLogger('tensorflow').setLevel(logging.ERROR)  

# suppress scientific notation

np.set_printoptions(suppress=True)

### Setup computation resources

# CPU / GPU /TPU

try: 
    # TPU Detection
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    # Connect to the TPUs
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)

    strategy = tf.distribute.TPUStrategy(cluster_resolver)
    print('\nRunning on TPU ', cluster_resolver.cluster_spec().as_dict()['worker'])
    print("All devices:", tf.config.list_logical_devices('TPU'))
except:
    physical_devices = tf.config.list_physical_devices('GPU') 
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        strategy = tf.distribute.MirroredStrategy()
        print('\nRunning on GPU...')
    else:
        strategy = tf.distribute.get_strategy()
        print('\nRunning on CPU...')
        
# RAM     

from psutil import virtual_memory

RAM_GB = virtual_memory().total / 1e9
print('\nYour runtime has {:.1f} gigabytes of available RAM\n'.format(RAM_GB))

if RAM_GB < 20:
    print('Not using a high-RAM runtime')
else:
    print('You are using a high-RAM runtime!')
