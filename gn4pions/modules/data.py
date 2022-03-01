# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_data.ipynb (unless otherwise specified).

__all__ = ['GraphDataGenerator']

# Cell
import numpy as np
import glob
import os
import uproot as ur
import time
from multiprocessing import Process, Queue, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
import random
import itertools

import pandas as pd

# Cell
class GraphDataGenerator:
    """
    DataGenerator class for extracting and formating data from list of root files
    This data generator uses the cell_geo file to create the input graph structure
    """
    def __init__(self,
                 pi0_file_list: list,
                 pion_file_list: list,
                 cellGeo_file: str,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs = 32,
                 preprocess = False,
                 output_dir = None):
        """Initialization"""

        self.preprocess = preprocess
        self.output_dir = output_dir

        if self.preprocess and self.output_dir is not None:
            self.pi0_file_list = pi0_file_list
            self.pion_file_list = pion_file_list
            assert len(pi0_file_list) == len(pion_file_list)
            self.num_files = len(self.pi0_file_list)
        else:
            self.file_list = pi0_file_list
            self.num_files = len(self.file_list)

        self.cellGeo_file = cellGeo_file

        self.cellGeo_data = ur.open(self.cellGeo_file)['CellGeo']
        self.geoFeatureNames = self.cellGeo_data.keys()[1:9]
        self.nodeFeatureNames = ['cluster_cell_E', *self.geoFeatureNames[:-2]]
        self.edgeFeatureNames = self.cellGeo_data.keys()[9:]
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_edgeFeatures = len(self.edgeFeatureNames)
        self.cellGeo_data = self.cellGeo_data.arrays(library='np')
        self.cellGeo_ID = self.cellGeo_data['cell_geo_ID'][0]
        self.sorter = np.argsort(self.cellGeo_ID)
        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle: np.random.shuffle(self.file_list)

        self.num_procs = np.min([num_procs, self.num_files])
        self.procs = []

        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

#         print(self.nodeFeatureNames)
#         print(self.edgeFeatureNames)

    def get_cluster_calib(self, event_data, event_ind, cluster_ind):
        """ Reading cluster calibration energy """

        cluster_calib_E = event_data['cluster_ENG_CALIB_TOT'][event_ind][cluster_ind]

        if cluster_calib_E <= 0:
            return None

        return np.log10(cluster_calib_E)

    def get_nodes(self, event_data, event_ind, cluster_ind):
        """ Reading Node features """

        cell_IDs = event_data['cluster_cell_ID'][event_ind][cluster_ind]
        cell_IDmap = self.sorter[np.searchsorted(self.cellGeo_ID, cell_IDs, sorter=self.sorter)]

        nodes = np.log10(event_data['cluster_cell_E'][event_ind][cluster_ind])
#         print('cluster_E', event_data['cluster_E'][event_ind][cluster_ind])
#         print('truthPartE', event_data['truthPartE'][event_ind][0])
        global_node = np.log10(event_data['truthPartE'][event_ind][0]) # prev: cluster_E (with cluster index)

        # Scaling the cell_geo_sampling by 28
        nodes = np.append(nodes, self.cellGeo_data['cell_geo_sampling'][0][cell_IDmap]/28.)
        for f in self.nodeFeatureNames[2:4]:
            nodes = np.append(nodes, self.cellGeo_data[f][0][cell_IDmap])
        # Scaling the cell_geo_rPerp by 3000
        nodes = np.append(nodes, self.cellGeo_data['cell_geo_rPerp'][0][cell_IDmap]/3000.)
        for f in self.nodeFeatureNames[5:]:
            nodes = np.append(nodes, self.cellGeo_data[f][0][cell_IDmap])

        nodes = np.reshape(nodes, (len(self.nodeFeatureNames), -1)).T
        cluster_num_nodes = len(nodes)

        # add dummy placeholder nodes for track features (not used in cluster cell nodes)
        nodes = np.hstack((nodes, np.zeros((cluster_num_nodes, 4))))

        return nodes, np.array([global_node]), cluster_num_nodes, cell_IDmap



    # WIP ----------------------------------------------------------------



    def get_track_node(self, event_data, event_index, track_index):
        """
        Creates node features for tracks
        Inputs:

        Returns:
            1 Dimensional array of node features for a single node
                NOTE the cluster get_node function is a 2D array of multiple nodes
                This function is used in a for loop so the end result is a 2D array
        """
        node_features = np.array(event_data["trackPt"][event_index][track_index])
        node_features = np.append(node_features, event_data["trackZ0"][event_index][track_index])
        node_features = np.append(node_features, event_data["trackEta_EMB2"][event_index][track_index])
        node_features = np.append(node_features, event_data["trackPhi_EMB2"][event_index][track_index])
        node_features = np.reshape(node_features, (len(node_features))).T

        # add dummy placeholder nodes for track features (not used in track cell nodes)
        node_features = np.hstack((np.zeros(7), node_features))

        return node_features

    def get_track_edges(self, num_track_nodes, start_index):
        """
        Creates the edge senders and recievers and edge features
        Inputs:
        (int) num_track_nodes: number of track nodes
        (int) start_index: the index of senders/recievers to start with. We should start with num_cluster_edges+1 to avoid overlap

        Returns:
        (np.array) edge_features:
        (np.array) senders:
        (np.array) recievers:
        """
        # Full Connected tracks
        # since we are fully connected, the order of senders and recievers doesn't matter
        # we just need to count each node - edges will have a placeholder feature
        connections = list(itertools.permutations(range(start_index, start_index + num_track_nodes),2))
        for i in range(5):
            connections.append((i, i))

        senders = np.array([x[0] for x in connections])
        recievers = np.array([x[0] for x in connections])
        edge_features = np.zeros((len(connections), 10))

        return senders, recievers, edge_features



    # end WIP ----------------------------------------------------------------



    def get_edges(self, cluster_num_nodes, cell_IDmap):
        """
        Reading edge features
        Returns senders, receivers, and edges
        """

        edge_inds = np.zeros((cluster_num_nodes, self.num_edgeFeatures))
        for i, f in enumerate(self.edgeFeatureNames):
            edge_inds[:, i] = self.cellGeo_data[f][0][cell_IDmap]
        edge_inds[np.logical_not(np.isin(edge_inds, cell_IDmap))] = np.nan

        senders, edge_on_inds = np.isin(edge_inds, cell_IDmap).nonzero()
        cluster_num_edges = len(senders)
        edges = np.zeros((cluster_num_edges, self.num_edgeFeatures))
        edges[np.arange(cluster_num_edges), edge_on_inds] = 1

        cell_IDmap_sorter = np.argsort(cell_IDmap)
        rank = np.searchsorted(cell_IDmap, edge_inds , sorter=cell_IDmap_sorter)
        receivers = cell_IDmap_sorter[rank[rank!=cluster_num_nodes]]

        return senders, receivers, edges

    def preprocessor(self, worker_id):
        """
        Prerocessing root file data for faster data
        generation during multiple training epochs
        """
        file_num = worker_id
        while file_num < self.num_files:
            print(f"Processing file number {file_num}")
            file = self.pion_file_list[file_num]
            event_data = np.load(file, allow_pickle=True).item()
            num_events = len(event_data[[key for key in event_data.keys()][0]])

            preprocessed_data = []

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]

                for i in range(num_clusters):
                    cluster_calib_E = self.get_cluster_calib(event_data, event_ind, i)

                    if cluster_calib_E is None:
                        continue

                    nodes, global_node, cluster_num_nodes, cell_IDmap = self.get_nodes(event_data, event_ind, i)
                    senders, receivers, edges = self.get_edges(cluster_num_nodes, cell_IDmap)


# WIP add track nodes and edges ----------------------------------------------------------------
                    track_nodes = np.empty((0, 11))
                    num_tracks = event_data['nTrack'][event_ind]
                    for track_index in range(num_tracks):
                        np.append(track_nodes, self.get_track_node(event_data, event_ind, track_index).reshape(1, -1), axis=0)

                    track_senders, track_receivers, track_edge_features = self.get_track_edges(len(track_nodes), cluster_num_nodes)

                    # append on the track nodes and edges to the cluster ones
                    nodes = np.append(nodes, np.array(track_nodes), axis=0)
                    edges = np.append(edges, track_edge_features, axis=0)
                    senders = np.append(senders, track_senders, axis=0)
                    receivers = np.append(receivers, track_receivers, axis=0)

                    # end WIP ----------------------------------------------------------------

                    graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                        'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                        'edges': edges.astype(np.float32)}
                    target = np.reshape([cluster_calib_E.astype(np.float32), 1], [1,2])

                    preprocessed_data.append((graph, target))

            file = self.pi0_file_list[file_num]
            event_data = np.load(file, allow_pickle=True).item()
            num_events = len(event_data[[key for key in event_data.keys()][0]])

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]

                for i in range(num_clusters):
                    cluster_calib_E = self.get_cluster_calib(event_data, event_ind, i)

                    if cluster_calib_E is None:
                        continue

                    nodes, global_node, cluster_num_nodes, cell_IDmap = self.get_nodes(event_data, event_ind, i)
                    senders, receivers, edges = self.get_edges(cluster_num_nodes, cell_IDmap)

                    # WIP add track nodes and edges ----------------------------------------------------------------
                    track_nodes = np.empty((0, 11))
                    num_tracks = event_data['nTrack'][event_ind]
                    for track_index in range(num_tracks):
                        np.append(track_nodes, self.get_track_node(event_data, event_ind, track_index).reshape(1, -1), axis=0)

                    track_senders, track_receivers, track_edge_features = self.get_track_edges(len(track_nodes), cluster_num_nodes)

                    nodes = np.append(nodes, np.array(track_nodes), axis=0)
                    edges = np.append(edges, track_edge_features, axis=0)
                    senders = np.append(senders, track_senders, axis=0)
                    receivers = np.append(receivers, track_receivers, axis=0)

                    # end WIP ----------------------------------------------------------------

                    graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                        'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                        'edges': edges.astype(np.float32)}
                    target = np.reshape([cluster_calib_E.astype(np.float32), 0], [1,2])

                    preprocessed_data.append((graph, target))

            random.shuffle(preprocessed_data)

            pickle.dump(preprocessed_data, open(self.output_dir + f'data_{file_num:03d}.p', 'wb'), compression='gzip')

            print(f"Finished processing {file_num} files")
            file_num += self.num_procs

    def preprocess_data(self):
        print('\nPreprocessing and saving data to {}'.format(self.output_dir))
        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)

        for p in self.procs:
            p.join()

        self.file_list = [self.output_dir + f'data_{i:03d}.p' for i in range(self.num_files)]

    def preprocessed_worker(self, worker_id, batch_queue):
        batch_graphs = []
        batch_targets = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])

                if len(batch_graphs) == self.batch_size:
                    batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)

                    batch_queue.put((batch_graphs, batch_targets))

                    batch_graphs = []
                    batch_targets = []

            file_num += self.num_procs

        if len(batch_graphs) > 0:
            batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)

            batch_queue.put((batch_graphs, batch_targets))

    def worker(self, worker_id, batch_queue):
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('Preprocessing is required for combined classification/regression models.')

    def check_procs(self):
        for p in self.procs:
            if p.is_alive(): return True

        return False

    def kill_procs(self):
        for p in self.procs:
            p.kill()

        self.procs = []

    def generator(self):
        """
        Generator that returns processed batches during training
        """
        batch_queue = Queue(2 * self.num_procs)

        for i in range(self.num_procs):
            p = Process(target=self.worker, args=(i, batch_queue), daemon=True)
            p.start()
            self.procs.append(p)

        while self.check_procs() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.0001)
            except:
                continue

            yield batch

        for p in self.procs:
            p.join()