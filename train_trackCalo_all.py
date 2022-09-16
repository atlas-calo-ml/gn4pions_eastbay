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

from gn4pions.modules.data_trackMultiCalo import MultiCaloTrackDataGenerator
import gn4pions.modules.models as models
sns.set_context('poster')

os.sys.path.append('../graph_nets/graph_nets/')
from utils_tf import fully_connect_graph_dynamic

LR_EPOCH = 20

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='gn4pions/configs/trackMultiCalo_regress.yaml')
    parser.add_argument('--restart', default=None)
    args = parser.parse_args()

    restart = args.restart
    if restart:
        config_file = restart + '/config.yaml'
        config = yaml.load(open(config_file))
        saved_losses = np.load(restart+'/losses.npz')
        training_loss_epoch = saved_losses['training'].tolist()
        val_loss_epoch = saved_losses['validation'].tolist()
        start_epoch = len(training_loss_epoch)

    else:
        config_file = args.config
        config = yaml.load(open(config_file))
        training_loss_epoch = []
        val_loss_epoch = []
        start_epoch = 0

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']

    data_dir = data_config['data_dir']
    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    batch_size = data_config['batch_size']
    n_clusters = data_config['n_clusters']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    already_preprocessed = True if restart else data_config['already_preprocessed']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']/(2**(start_epoch//LR_EPOCH))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['gpu'])
    log_freq = train_config['log_freq']
    if start_epoch:
        save_dir = restart
    else:
        save_dir = train_config['save_dir'] + '/Block_'+time.strftime("%Y%m%d_%H%M")+'_'+args.config.replace('.yaml','').split('/')[-1]
        os.makedirs(save_dir, exist_ok=True)
        yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    logging.basicConfig(level=logging.INFO, 
                        format='%(message)s', 
                        filename=save_dir + '/output.log')

    if start_epoch:
        logging.info('\n\nRestarting traning from: {}, epoch: {}'.format(save_dir, start_epoch))

    logging.info('Using config file {}'.format(config_file))
    # logging.info('Running training for {} with concant_input: {}\n'.format(particle_type, concat_input))

    pion_files = np.sort(glob.glob(data_dir+'/*npy'))
    train_start = 0
    train_end = train_start + num_train_files
    val_end = train_end + num_val_files
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

            pion_train_files = train_files
            pion_val_files = val_files

            train_output_dir = None
            val_output_dir = None

    cell_geo_file = '/usr/workspace/hip/ML4Jets/regression_images/graph_examples/cell_geo.root'

    data_gen_train = MultiCaloTrackDataGenerator(pion_file_list=pion_train_files,
                                                 cellGeo_file=cell_geo_file,
                                                 batch_size=batch_size,
                                                 n_clusters=n_clusters,
                                                 shuffle=shuffle,
                                                 num_procs=num_procs,
                                                 preprocess=preprocess,
                                                 output_dir=train_output_dir)

    data_gen_val = MultiCaloTrackDataGenerator(pion_file_list=pion_val_files,
                                               cellGeo_file=cell_geo_file,
                                               batch_size=batch_size,
                                               n_clusters=n_clusters,
                                               shuffle=shuffle,
                                               num_procs=num_procs,
                                               preprocess=preprocess,
                                               output_dir=val_output_dir)

    # if preprocess and not already_preprocessed:
    #     exit()

    # Optimizer.
    #optimizer = snt.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    logging.info('\nLearning rate set to: {:.5e}'.format(optimizer.learning_rate.value()))
    print('\nLearning rate set to: {:.5e}'.format(optimizer.learning_rate.value()))

    model = models.MultiOutBlockModel(global_output_size=1, num_outputs=1, model_config=model_config)

    checkpoint = tf.train.Checkpoint(module=model)
    best_ckpt_prefix = os.path.join(save_dir, 'best_model')
    best_ckpt = tf.train.latest_checkpoint(save_dir)
    last_ckpt_path = save_dir + '/last_saved_model'
    if best_ckpt is not None:
        checkpoint.restore(best_ckpt)
    if os.path.exists(last_ckpt_path+'.index'):
        checkpoint.read(last_ckpt_path)
    else:
        checkpoint.write(last_ckpt_path)

    def convert_to_tuple(graphs):
        nodes = []
        edges = []
        global_nodes = []
        senders = []
        receivers = []
        n_node = []
        n_edge = []
        offset = 0

        for graph in graphs:
            nodes.append(graph['nodes'])
            edges.append(graph['edges'])
            global_nodes.append([graph['globals']])
            senders.append(graph['senders'] + offset)
            receivers.append(graph['receivers'] + offset)
            n_node.append(graph['nodes'].shape[:1])
            n_edge.append(graph['edges'].shape[:1])

            offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes))
        edges = tf.convert_to_tensor(np.concatenate(edges), dtype=tf.float32)
        global_nodes = tf.convert_to_tensor(np.concatenate(global_nodes))
        senders = tf.convert_to_tensor(np.concatenate(senders))
        receivers = tf.convert_to_tensor(np.concatenate(receivers))
        n_node = tf.convert_to_tensor(np.concatenate(n_node))
        n_edge = tf.convert_to_tensor(np.concatenate(n_edge))

        graph = GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=global_nodes,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge
            )
        
        return graph
       
    def get_batch(data_iter):
        for graphs, targets, track_meta, cluster_meta in data_iter:
            graphs = convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets)
            
            yield graphs, targets, track_meta, cluster_meta

    samp_graph, samp_target, _, _ = next(get_batch(data_gen_train.generator()))
    data_gen_train.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)
    
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    # mse_loss = tf.keras.losses.MeanSquaredError()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_fn(targets, regress_preds):
        regress_loss = mae_loss(targets, regress_preds)
        # class_loss = bce_loss(targets[:,1:], class_preds)
        # combined_loss = alpha*regress_loss + (1 - alpha)*class_loss 
        return regress_loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            regress_output = model(graphs)[0]
            regress_preds = regress_output.globals
            # class_preds = class_output.globals
            loss = loss_fn(targets, regress_preds)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
    def val_step(graphs, targets):
        regress_output = model(graphs)[0]
        regress_preds = regress_output.globals
        # class_preds = class_output.globals
        loss = loss_fn(targets, regress_preds)

        return loss, regress_preds

    curr_loss = np.min(np.mean(val_loss_epoch, axis=1)) if start_epoch else 1e5
    for e in range(start_epoch, epochs):

        logging.info('\nStarting epoch: {}'.format(e))
        print('\nStarting epoch: {}'.format(e))
        epoch_start = time.time()

        training_loss = []
        val_loss = []

        # Train
        logging.info('Training...')
        i = 1
        start = time.time()
        for graph_data_tr, targets_tr, _, _ in get_batch(data_gen_train.generator()):#train_iter):
            #if i==1:
            losses_tr = train_step(graph_data_tr, targets_tr)
            training_loss.append(losses_tr.numpy())

            if not (i-1)%log_freq:
                end = time.time()
                logging.info('Iter: {:04d}, Tr_loss_mean: {:.6f}, Took {:.3f}secs'. \
                      format(i, 
                             np.mean(training_loss), 
                             end-start))
                start = time.time()
            
            i += 1 

        end = time.time()
        logging.info('Iter: {:04d}, Tr_loss_mean: {:.6f}, Took {:.3f}secs'. \
              format(i, 
                     np.mean(training_loss), 
                     end-start))
        training_loss_epoch.append(training_loss)
        training_end = time.time()

        # validate
        logging.info('\nValidation...')
        i = 1
        all_targets = []
        all_outputs = []
        track_meta_cols = data_gen_val.track_feature_names
        cluster_meta_cols = data_gen_val.cluster_feature_names

        df_track = pd.DataFrame(columns=track_meta_cols)
        # df_cluster = pd.DataFrame(columns=cluster_meta_cols)

        start = time.time()
        for graph_data_val, targets_val, track_meta_val, cluster_meta_val in get_batch(data_gen_val.generator()):#val_iter):
            losses_val, regress_vals = val_step(graph_data_val, targets_val)

            targets_val = targets_val.numpy()
            regress_vals = regress_vals.numpy()

            targets_val = 10**targets_val
            regress_vals = 10**regress_vals

            val_loss.append(losses_val.numpy())

            all_targets.append(targets_val)
            all_outputs.append(regress_vals)

            track_meta_val = np.array(track_meta_val)
            df_track = df_track.append(pd.DataFrame(track_meta_val.squeeze(), columns=track_meta_cols))
            # cluster_meta_val = np.array(cluster_meta_val)
            # df_cluster = df_cluster.append(pd.DataFrame(cluster_meta_val.squeeze(), columns=cluster_meta_cols))

            if not (i-1)%log_freq:
                end = time.time()
                logging.info('Iter: {:04d}, Val_loss_mean: {:.6f}, Took {:.3f}secs'. \
                      format(i, 
                             np.mean(val_loss), 
                             end-start))
                start = time.time()
            
            i += 1

        end = time.time()
        logging.info('Iter: {:04d}, Val_loss_mean: {:.6f}, Took {:.3f}secs'. \
              format(i, 
                     np.mean(val_loss), 
                     end-start))
        epoch_end = time.time()

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        val_loss_epoch.append(val_loss)
    
        np.savez(save_dir+'/losses', 
                training=training_loss_epoch, validation=val_loss_epoch,
                )
        checkpoint.write(last_ckpt_path)

        val_mins = int((epoch_end - training_end)/60)
        val_secs = int((epoch_end - training_end)%60)
        training_mins = int((training_end - epoch_start)/60)
        training_secs = int((training_end - epoch_start)%60)

        logging.info('\nEpoch {} ended\nTraining: {:2d}:{:02d}\nValidation: {:2d}:{:02d}'. \
                     format(e, training_mins, training_secs, val_mins, val_secs))
        print('\nEpoch {} ended\nTraining: {:2d}:{:02d}\nValidation: {:2d}:{:02d}'. \
              format(e, training_mins, training_secs, val_mins, val_secs))

        if np.mean(val_loss)<curr_loss:
            logging.info('\nLoss decreased from {:.6f} to {:.6f}'.format(curr_loss, np.mean(val_loss)))
            logging.info('Checkpointing and saving predictions to:\n{}'.format(save_dir))
            print('\nLoss decreased from {:.6f} to {:.6f}'.format(curr_loss, np.mean(val_loss)))
            print('Checkpointing and saving predictions to:\n{}'.format(save_dir))
            curr_loss = np.mean(val_loss)
            np.savez(save_dir+'/predictions', 
                    targets=all_targets, 
                    outputs=all_outputs)
            checkpoint.save(best_ckpt_prefix)
            df_track.to_pickle(save_dir+'/track_meta_df.pkl')
            # df_cluster.to_pickle(save_dir+'/cluster_meta_df.pkl')
        else: 
            logging.info('\nLoss didnt decrease from {:.6f}'.format(curr_loss))
            print('\nLoss didnt decrease from {:.6f}'.format(curr_loss))

        if (not (e+1)%LR_EPOCH) and optimizer.learning_rate>1e-6:
            optimizer.learning_rate = optimizer.learning_rate/2
            logging.info('\nLearning rate decreased to: {:.5e}'.format(optimizer.learning_rate.value()))
            print('\nLearning rate decreased to: {:.5e}'.format(optimizer.learning_rate.value()))

