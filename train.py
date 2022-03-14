#!/usr/bin/env python
# coding: utf-8

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", default=None, type=str, help="Specify training config file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Loading model config
    config_file = args.config
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)

    # Data config
    data_config = config['data']
    data_dir = data_config['data_dir']
    cell_geo_file = data_config['cell_geo_file']
    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    already_preprocessed = data_config['already_preprocessed']

    # Model Config
    model_config = config['model']
    concat_input = model_config['concat_input']

    # Training config
    train_config = config['training']
    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    alpha = train_config['alpha']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['gpu'])
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    log_freq = train_config['log_freq']
    save_dir = train_config['save_dir'] + config_file.replace('.yaml','').split('/')[-1] + '_' + time.strftime("%Y%m%d")

    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    # Read data and create data generators
    pi0_files = np.sort(glob.glob(data_dir+'pi0_files/*.npy'))
    pion_files = np.sort(glob.glob(data_dir+'pion_files/*.npy'))

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

            pi0_train_files = train_files
            pi0_val_files = val_files
            pion_train_files = None
            pion_val_files = None

            train_output_dir = None
            val_output_dir = None

    # Training Data Generator
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

    # Get batch of data
    def get_batch(data_iter):
        for graphs, targets in data_iter:
            graphs, energies, etas, em_probs, lc_es = convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets)
            yield graphs, targets, energies, etas, em_probs, lc_es

    # Define loss function        
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    def loss_fn(targets, regress_preds, class_preds):
        regress_loss = mae_loss(targets[:,:1], regress_preds)
        class_loss = bce_loss(targets[:,1:], class_preds)
        combined_loss = alpha*regress_loss + (1 - alpha)*class_loss 
        return regress_loss, class_loss, combined_loss

    # Get a sample graph for tf.function decorator
    samp_graph, samp_target, samp_e, samp_eta, samp_em_prob, samp_lc_e = next(get_batch(data_gen_train.generator()))
    data_gen_train.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)

    # Training set
    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            regress_output, class_output = model(graphs)
            regress_preds = regress_output.globals
            class_preds = class_output.globals
            regress_loss, class_loss, loss = loss_fn(targets, regress_preds, class_preds)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return regress_loss, class_loss, loss

    # Validation Step
    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def val_step(graphs, targets):
        regress_output, class_output = model(graphs)
        regress_preds = regress_output.globals
        class_preds = class_output.globals
        regress_loss, class_loss, loss = loss_fn(targets, regress_preds, class_preds)
        return regress_loss, class_loss, loss, regress_preds, class_preds

    # Model 
    model = MultiOutWeightedRegressModel(global_output_size=1, num_outputs=2, model_config=model_config)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Average epoch losses
    training_loss_epoch = []
    training_loss_regress_epoch = []
    training_loss_class_epoch = []
    val_loss_epoch = []
    val_loss_regress_epoch = []
    val_loss_class_epoch = []

    # Model checkpointing, load latest model if available
    checkpoint = tf.train.Checkpoint(module=model)
    checkpoint_prefix = os.path.join(save_dir, 'latest_model')
    latest = tf.train.latest_checkpoint(save_dir)
    if latest is not None:
        checkpoint.restore(latest)
    else:
        checkpoint.save(checkpoint_prefix)

    # Run training
    curr_loss = 1e5

    for e in range(epochs):

        print(f'\n\nStarting epoch: {e}')
        epoch_start = time.time()

        # Batchwise losses
        training_loss = []
        training_loss_regress = []
        training_loss_class = []
        val_loss = []
        val_loss_regress = []
        val_loss_class = []

        # Train
        print('Training...')
        start = time.time()
        for i, (graph_data_tr, targets_tr, energies_tr, etas_tr, em_probs_tr, lc_es_tr) in enumerate(get_batch(data_gen_train.generator())):
            losses_tr_rg, losses_tr_cl, losses_tr = train_step(graph_data_tr, targets_tr)

            training_loss.append(losses_tr.numpy())
            training_loss_regress.append(losses_tr_rg.numpy())
            training_loss_class.append(losses_tr_cl.numpy())

            if not (i-1)%log_freq:
                end = time.time()
                print(f'Iter: {i:04d}, ', end='')
                print(f'Tr_loss_mean: {np.mean(training_loss):.4f}, ', end='')
                print(f'Tr_loss_rg_mean: {np.mean(training_loss_regress):.4f}, ', end='') 
                print(f'Tr_loss_cl_mean: {np.mean(training_loss_class):.4f}, ', end='') 
                print(f'Took {end-start:.4f}secs')
                start = time.time()

        training_loss_epoch.append(training_loss)
        training_loss_regress_epoch.append(training_loss_regress)
        training_loss_class_epoch.append(training_loss_class)
        training_end = time.time()

        # validate
        print('\nValidation...')
        all_targets = []
        all_outputs = []
        all_energies = []
        all_etas = []
        all_em_probs = []
        all_lc_es = []
        start = time.time()
        for i, (graph_data_val, targets_val, energies_val, etas_val, em_probs_val, lc_es_val) in enumerate(get_batch(data_gen_val.generator())):
            losses_val_rg, losses_val_cl, losses_val, regress_vals, class_vals = val_step(graph_data_val, targets_val)

            targets_val = targets_val.numpy()
            regress_vals = regress_vals.numpy()
            class_vals = class_vals.numpy()

            ### These variables are stored as log_10, so need to exponentiate them again here 
            targets_val[:,0] = 10**targets_val[:,0]
            regress_vals = 10**regress_vals
            class_vals =  tf.math.sigmoid(class_vals)
            energy = 10**graph_data_val.globals 

            output_vals = np.hstack([regress_vals, class_vals])

            val_loss.append(losses_val.numpy())
            val_loss_regress.append(losses_val_rg.numpy())
            val_loss_class.append(losses_val_cl.numpy())

            all_targets.append(targets_val)
            all_outputs.append(output_vals)
            all_energies.append([10**energy for energy in energies_val])
            all_etas.append(etas_val)
            all_em_probs.append(em_probs_val)
            all_lc_es.append([10**energy for energy in lc_es_val])

            if not (i-1)%log_freq:
                end = time.time()
                print(f'Iter: {i:04d}, ', end='')
                print(f'Val_loss_mean: {np.mean(val_loss):.4f}, ', end='')
                print(f'Val_loss_rg_mean: {np.mean(val_loss_regress):.4f}, ', end='') 
                print(f'Val_loss_cl_mean: {np.mean(val_loss_class):.4f}, ', end='') 
                print(f'Took {end-start:.4f}secs')
                start = time.time()

        epoch_end = time.time()

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        all_energies = np.concatenate(all_energies)
        all_etas = np.concatenate(all_etas)
        all_em_probs = np.concatenate(all_em_probs)
        all_lc_es = np.concatenate(all_lc_es)

        val_loss_epoch.append(val_loss)
        val_loss_regress_epoch.append(val_loss_regress)
        val_loss_class_epoch.append(val_loss_class)

        # Book keeping
        val_mins = int((epoch_end - training_end)/60)
        val_secs = int((epoch_end - training_end)%60)
        training_mins = int((training_end - epoch_start)/60)
        training_secs = int((training_end - epoch_start)%60)
        print(f'\nEpoch {e} ended')
        print(f'Training: {training_mins:2d}:{training_secs:02d}')
        print(f'Validation: {val_mins:2d}:{val_secs:02d}')


        # Save losses
        np.savez(save_dir+'/losses', 
                training=training_loss_epoch, validation=val_loss_epoch,
                training_regress=training_loss_regress_epoch, validation_regress=val_loss_regress_epoch,
                training_class=training_loss_class_epoch, validation_class=val_loss_class_epoch,
                )


        # Checkpoint if validation loss improved
        if np.mean(val_loss)<curr_loss:
            print(f'Loss decreased from {curr_loss:.4f} to {np.mean(val_loss):.4f}')
            print(f'Checkpointing and saving predictions to:\n{save_dir}')
            curr_loss = np.mean(val_loss)
            np.savez(save_dir+'/predictions', 
                    targets=all_targets, 
                    outputs=all_outputs,
                    energies=all_energies,
                    etas=all_etas,
                    em_probs=all_em_probs,
                    lc_es=all_lc_es)
            checkpoint.save(checkpoint_prefix)
        else: 
            print(f'Loss didnt decrease from {curr_loss:.4f}')

        # Decrease learning rate every few epochs
        if not (e+1)%2:   #%20:
            optimizer.learning_rate = optimizer.learning_rate/10
            print(f'Learning rate decreased to: {optimizer.learning_rate.value():.3e}')

