import numpy as np
import os
import sys
import glob
import uproot as ur
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tensorflow as tf
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
import sonnet as snt
import argparse
import yaml
import logging
import tensorflow as tf
from tqdm import tqdm

from gn4pions.modules.data import GraphDataGenerator
from gn4pions.modules.models import MultiOutWeightedRegressModel
from gn4pions.modules.utils import convert_to_tuple

sns.set_context('poster')

# Loading model config
config_file = '/hpcfs/users/a1768536/AGPF/gnn4pions/gn4pions_eastbay/gn4pions/configs/test.yaml' # for a quick run of the notebook
# config_file = 'gn4pions/configs/baseline.yaml' # for actual training
config = yaml.load(open(config_file), Loader=yaml.FullLoader)

# Data config
data_config = config['data']
cell_geo_file = data_config['cell_geo_file']
data_dir = data_config['data_dir']
num_train_files = data_config['num_train_files']
num_val_files = data_config['num_val_files']
batch_size = data_config['batch_size']
shuffle = data_config['shuffle']
num_procs = data_config['num_procs']
preprocess = data_config['preprocess']
output_dir = data_config['output_dir']
already_preprocessed = False
# already_preprocessed = data_config['already_preprocessed']  # Set to false when running training for first time

# Model Config
model_config = config['model']
concat_input = model_config['concat_input']

# Traning Config
train_config = config['training']

epochs = train_config['epochs']
learning_rate = train_config['learning_rate']
alpha = train_config['alpha']
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
physical_devices = tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
log_freq = train_config['log_freq']
save_dir = train_config['save_dir'] + config_file.replace('.yaml','').split('/')[-1] + '_' + time.strftime("%Y%m%d")

os.makedirs(save_dir, exist_ok=True)
yaml.dump(config, open(save_dir + '/config.yaml', 'w'))



### Import the data ###

pi0_files = np.sort(glob.glob(data_dir+'pi0_files/*.npy'))
pion_files = np.sort(glob.glob(data_dir+'pion_files/*.npy'))

len(pi0_files)
len(pion_files)

train_start = 0
train_end = train_start + num_train_files
val_end = train_end + num_val_files

pi0_train_files = pi0_files[train_start:train_end]
pi0_val_files = pi0_files[train_end:val_end]
pion_train_files = pion_files[train_start:train_end]
pion_val_files = pion_files[train_end:val_end]

train_output_dir = None
val_output_dir = None




# Get Data
if preprocess:
    train_output_dir = output_dir + '/train/'
    val_output_dir = output_dir + '/val/'

    if already_preprocessed:
        train_files = np.sort(glob.glob(train_output_dir+'*.p'))[:num_train_files]
        val_files = np.sort(glob.glob(val_output_dir+'*.p'))[:num_val_files]

        pi0_train_files = None
        pi0_val_files = None
        pion_train_files = train_files
        pion_val_files = val_files

        train_output_dir = None
        val_output_dir = None

# Traning Data Generator
# Will preprocess data if it doesnt find pickled files
data_gen_train = GraphDataGenerator(pi0_file_list=pi0_train_files,
                                    pion_file_list=pion_train_files,
                                    cellGeo_file=cell_geo_file,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_procs=num_procs,
                                    preprocess=preprocess,
                                    output_dir=train_output_dir)

# Validation Data generator
# Will preprocess data if it doesnt find pickled files
data_gen_val = GraphDataGenerator(pi0_file_list=pi0_val_files,
                                  pion_file_list=pion_val_files,
                                  cellGeo_file=cell_geo_file,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_procs=num_procs,
                                  preprocess=preprocess,
                                  output_dir=val_output_dir)