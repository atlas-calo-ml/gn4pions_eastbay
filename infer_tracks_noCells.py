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

from gn4pions.modules.data_tracks import TrackGraphDataGenerator
import gn4pions.modules.models as models
sns.set_context('poster')

os.sys.path.append('../graph_nets/graph_nets/')
from utils_tf import fully_connect_graph_dynamic

LR_EPOCH = 15

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir',
                        default='results/onetrack_multicluster/Block_large_20220512_1927_track_regress_noCell_ncluster_deepset/')
    args = parser.parse_args()

    config = yaml.load(open(args.save_dir + '/config.yaml'))

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
    output_dir = '/p/vast1/karande1/heavyIon/data/preprocessed_data/infer/tracks/'
    already_preprocessed = data_config['already_preprocessed']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['gpu'])
    log_freq = train_config['log_freq']
    save_dir = args.save_dir

    logging.basicConfig(level=logging.INFO, 
                        format='%(message)s', 
                        filename=save_dir + '/infer_output.log')
    logging.info('Using config file from {}'.format(args.save_dir)) 

    pion_files = np.sort(glob.glob(data_dir+'/*npy'))
    val_start = 250
    pion_val_files = pion_files[val_start:]

    val_output_dir = None
            
    # Get Data
    if preprocess:
        val_output_dir = output_dir + '/test/'

        if already_preprocessed:
            val_files = np.sort(glob.glob(val_output_dir+'*.p'))# [:num_val_files]

            pion_val_files = val_files

            val_output_dir = None


    data_gen_val = TrackGraphDataGenerator(pion_file_list=pion_val_files,
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
            global_nodes.append(graph['globals'])
            senders.append(graph['senders'] + offset)
            receivers.append(graph['receivers'] + offset)
            n_node.append(graph['nodes'].shape[:1])
            n_edge.append(graph['edges'].shape[:1])

            offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes))
        edges = tf.convert_to_tensor(np.concatenate(edges))
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

    samp_graph, samp_target, _, _ = next(get_batch(data_gen_val.generator()))
    data_gen_val.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)
    
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_fn(targets, regress_preds):
        regress_loss = mae_loss(targets, regress_preds)
        # class_loss = bce_loss(targets[:,1:], class_preds)
        # combined_loss = alpha*regress_loss + (1 - alpha)*class_loss 
        return regress_loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,], dtype=tf.float32)])
    def val_step(graphs, targets):
        regress_output = model(graphs)[0]
        regress_preds = regress_output.globals
        # class_preds = class_output.globals
        loss = loss_fn(targets, regress_preds)

        return loss, regress_preds

    checkpoint = tf.train.Checkpoint(module=model)
    checkpoint_prefix = os.path.join(save_dir, 'best_model')
    latest = tf.train.latest_checkpoint(save_dir)
    logging.info(f'Restoring checkpoint from {latest}')
    print(f'Restoring checkpoint from {latest}')
    checkpoint.restore(latest)

    # validate
    logging.info('\nValidation...')
    i = 1
    all_targets = []
    all_outputs = []
    track_meta_cols = data_gen_val.track_feature_names
    cluster_meta_cols = data_gen_val.cluster_feature_names
    
    val_loss = []

    df_track = pd.DataFrame(columns=track_meta_cols)
    df_cluster = pd.DataFrame(columns=cluster_meta_cols)

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
            logging.info('Iter: {:04d}, Val_loss_mean: {:.4f}, Took {:.3f}secs'. \
                  format(i, 
                         np.mean(val_loss), 
                         end-start))
            start = time.time()
        
        i += 1

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    np.savez(save_dir+'/inference_predictions', 
            targets=all_targets, 
            outputs=all_outputs)
    df_track.to_pickle(save_dir+'/track_meta_df.pkl')
