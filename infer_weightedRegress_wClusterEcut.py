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
import pandas as pd

from gn4pions.modules.data_infer_wClusterEcuts import MPGraphDataGenerator
from gn4pions.modules.models import MultiOutWeightedRegressModel
from gn4pions.modules.utils import convert_to_tuple

sns.set_context('poster')

def get_batch(data_iter):
    for graphs, targets, meta in data_iter:
        graphs = convert_to_tuple(graphs)
        targets = tf.convert_to_tensor(targets)
        
        yield graphs, targets, meta

index_to_class = {0: 'pi0', 1: 'pion'}

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', default='results/Block_multiJob_20220218_simult_wClusterEcut/')
    args = parser.parse_args()

    config = yaml.load(open(args.save_dir + '/config.yaml'))

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']

    data_dir = data_config['data_dir']
    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    cluster_E_cut = data_config['cluster_E_cut']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = '/p/vast1/karande1/heavyIon/data/preprocessed_data/infer/geo_wClusterEcut/'

    already_preprocessed = data_config['already_preprocessed']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    alpha = train_config['alpha']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['gpu'])
    log_freq = train_config['log_freq']
    save_dir = args.save_dir

    logging.basicConfig(level=logging.INFO, 
                        format='%(message)s', 
                        filename=save_dir + '/infer_output.log')
    logging.info('Using config file from {}'.format(args.save_dir)) 
    pi0_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*pi0*/*root'))
    pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*pion*/*root'))
    train_start = 0
    train_end = train_start + num_train_files
    val_end = train_end + num_val_files
    pi0_val_files = pi0_files[train_end:val_end]
    pion_val_files = pion_files[train_end:val_end]

    val_output_dir = None
            
    # Get Data
    if preprocess:
        val_output_dir = output_dir + '/val/'

        if already_preprocessed:
            val_files = np.sort(glob.glob(val_output_dir+'*.p'))[:num_val_files]

            pi0_val_files = val_files
            pion_val_files = None

            val_output_dir = None

    data_gen_val = MPGraphDataGenerator(pi0_file_list=pi0_val_files,
                                        pion_file_list=pion_val_files,
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        cluster_E_cut=cluster_E_cut,
                                        num_procs=num_procs,
                                        preprocess=preprocess,
                                        output_dir=val_output_dir)

    # Optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = MultiOutWeightedRegressModel(global_output_size=1, num_outputs=2, model_config=model_config)

    mae_loss = tf.keras.losses.MeanAbsoluteError()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_fn(targets, regress_preds, class_preds):
        regress_loss = mae_loss(targets[:,:1], regress_preds)
        class_loss = bce_loss(targets[:,1:], class_preds)
        combined_loss = alpha*regress_loss + (1 - alpha)*class_loss 
        return regress_loss, class_loss, combined_loss

    samp_graph, samp_target, _ = next(get_batch(data_gen_val.generator()))
    data_gen_val.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)
    
    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def val_step(graphs, targets):
        regress_output, class_output = model(graphs)
        regress_preds = regress_output.globals
        class_preds = class_output.globals
        regress_loss, class_loss, loss = loss_fn(targets, regress_preds, class_preds)

        return regress_loss, class_loss, loss, regress_preds, class_preds

    val_loss = []
    val_loss_regress = []
    val_loss_class = []

    checkpoint = tf.train.Checkpoint(module=model)
    checkpoint_prefix = os.path.join(save_dir, 'best_model')
    latest = tf.train.latest_checkpoint(save_dir)
    logging.info(f'Restoring checkpoint from {latest}')
    print(f'Restoring checkpoint from {latest}')
    checkpoint.restore(latest)

    meta_cols = data_gen_val.meta_features
    meta_cols.extend(['pred_cluster_ENG_CALIB_TOT', 'pred_type', 'pred_prob'])

    # validate
    df = pd.DataFrame(columns=meta_cols)
    df.to_csv(save_dir+'/validation_data_predictions.csv', index=False)
    logging.info('\nStarting inference on validation data..')
    print('\nStarting inference on validation data..')
    i = 1
    start = time.time()
    for graph_data_val, targets_val, meta_vals in get_batch(data_gen_val.generator()):#val_iter):
        losses_val_rg, losses_val_cl, losses_val, regress_vals, class_vals = val_step(graph_data_val, targets_val)

        targets_val = targets_val.numpy()
        regress_vals = regress_vals.numpy()
        class_vals = class_vals.numpy()

        targets_val[:,0] = 10**targets_val[:,0]
        regress_vals = 10**regress_vals
        class_vals =  tf.math.sigmoid(class_vals)   # 1 / (1 + np.exp(class_vals))
        class_pred_vals = [index_to_class[int(c>.5)] for c in class_vals] 
        class_pred_vals = np.array([class_pred_vals]).reshape(-1, 1)
        class_vals = np.array([class_vals]).reshape(-1, 1)
       
        meta_vals = np.array(meta_vals)
        df = df.append(pd.DataFrame(np.hstack([meta_vals, regress_vals, class_pred_vals, class_vals]), 
                                          columns=meta_cols))

        val_loss.append(losses_val.numpy())
        val_loss_regress.append(losses_val_rg.numpy())
        val_loss_class.append(losses_val_cl.numpy())

        if not (i-1)%log_freq:
            end = time.time()
            logging.info('Iter: {:04d}, Val_loss_mean: {:.4f}, Val_loss_rg_mean: {:.4f}, Val_loss_cl_mean: {:.4f}, Took {:.3f}secs'. \
                  format(i, 
                         np.mean(val_loss), 
                         np.mean(val_loss_regress), 
                         np.mean(val_loss_class), 
                         end-start))
            print('Iter: {:04d}, Val_loss_mean: {:.4f}, Val_loss_rg_mean: {:.4f}, Val_loss_cl_mean: {:.4f}, Took {:.3f}secs'. \
                  format(i, 
                         np.mean(val_loss), 
                         np.mean(val_loss_regress), 
                         np.mean(val_loss_class), 
                         end-start))
            start = time.time()
            df.to_csv(save_dir+'/validation_data_predictions.csv', mode='a', index=False, header=False)
            df = pd.DataFrame(columns=meta_cols)
            
        i += 1

    df.to_csv(save_dir+'/validation_data_predictions.csv', mode='a', index=False, header=False)
