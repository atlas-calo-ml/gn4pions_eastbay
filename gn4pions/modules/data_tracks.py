# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_data_noCalirometer.ipynb (unless otherwise specified).

__all__ = ['TrackGraphDataGenerator']

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
class TrackGraphDataGenerator:
    """
    DataGenerator class for extracting and formating data from list of root files
    This data generator uses the cell_geo file to create the input graph structure
    """
    def __init__(self,
                 pion_file_list: list,
                 n_clusters: int,
                 batch_size: int,
                 shuffle: bool = True,
                 num_procs = 32,
                 preprocess = False,
                 output_dir = None):
        """Initialization"""

        self.preprocess = preprocess
        self.output_dir = output_dir

        if self.preprocess and self.output_dir is not None:
            self.pion_file_list = pion_file_list
            self.num_files = len(self.pion_file_list)
        else:
            self.file_list = pion_file_list
            self.num_files = len(self.file_list)

        self.nodeFeatureNames = ['cluster_E', 'track_pt', 'track_eta']
        self.num_nodeFeatures = len(self.nodeFeatureNames)

        self.track_feature_names = ['trackPt','trackD0','trackZ0', 'trackEta_EMB2','trackPhi_EMB2',
                                    'trackEta','trackPhi','truthPartE', 'truthPartPt']
        self.cluster_feature_names = ['cluster_E', 'cluster_Eta', 'cluster_Phi', 'cluster_ENG_CALIB_TOT',
                                      'cluster_EM_PROBABILITY','cluster_E_LCCalib','cluster_HAD_WEIGHT']

        self.dr_thresh = 1.2
        self.clusterThresh = .5

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_clusters = n_clusters

        if self.shuffle: np.random.shuffle(self.file_list)

        self.num_procs = np.min([num_procs, self.num_files])
        self.procs = []

        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

    def get_meta(self, event_data, event_ind, c_inds):
        """
        Reading meta data
        """
        track_meta_data = []
        for f in self.track_feature_names:
            track_meta_data.append(event_data[f][event_ind])

        cluster_meta_data = []
        for c in c_inds:
            curr_meta = []

            for f in self.cluster_feature_names:
                curr_meta.append(event_data[f][event_ind][c])

            cluster_meta_data.append(curr_meta)

        return np.array(track_meta_data, dtype=np.float32), np.array(cluster_meta_data, dtype=np.float32)

    def get_nodes(self, event_data, event_ind, c_inds):
        """ Reading Node features """

        nodes = []
        for c in c_inds:
            cluster_E = np.log10(event_data['cluster_E'][event_ind][c])
            curr_node = [cluster_E, 0, 0]
            nodes.append(curr_node)

        # add the track node
        trackPt = np.log10(event_data['trackPt'][event_ind][0])
        nodes.append([0, trackPt, event_data['trackEta'][event_ind][0]])

        return np.array(nodes, dtype=np.float32)

    def get_cluster_inds(self, event_data, event_ind):

        if self.n_clusters==-1:   # get all nodes satisfying dR criterion
            c_inds = range(event_data['nCluster'][event_ind])
            c_inds = [c for c in c_inds if (event_data['dR'][event_ind][c]<self.dr_thresh) and
                      (event_data['cluster_E'][event_ind][c]>self.clusterThresh)]
        else:                # get n leading nodes satisfying dR criterion
            c_inds = np.argsort(event_data['cluster_E'][event_ind])[::-1]
            c_inds = [c for c in c_inds if (event_data['dR'][event_ind][c]<self.dr_thresh) and
                      (event_data['cluster_E'][event_ind][c]>self.clusterThresh)]
            c_inds = c_inds[:self.n_clusters]

        return c_inds

    def preprocessor(self, worker_id):
        """
        Prerocessing root file data for faster data
        generation during multiple training epochs
        """
        file_num = worker_id
        while file_num < self.num_files:
            print(f"Processing file number {file_num}")

            ### Pions
            if len(self.pion_file_list) == 0:
                print("No pion files.")
            elif len(self.pion_file_list) > 0:
                file = self.pion_file_list[file_num]
                event_data = np.load(file, allow_pickle=True).item()
                num_events = len(event_data[[key for key in event_data.keys()][0]])

                preprocessed_data = []

                for event_ind in range(num_events):
                    truth_particle_E = np.log10(event_data['truthPartE'][event_ind][0]) # first one is the pion!
                    trackPt = event_data['trackPt'][event_ind][0]
                    if trackPt>5000:
                        continue

                    c_inds = self.get_cluster_inds(event_data, event_ind)
                    if not len(c_inds):
                        continue
                    nodes = self.get_nodes(event_data, event_ind, c_inds)
                    num_nodes = len(nodes)
                    senders = [i for i in range(num_nodes) for j in range(num_nodes) if i != j]
                    receivers = [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
                    n_edges = len(senders)
                    edges = np.zeros(shape=[n_edges, 0], dtype=np.float32)
                    global_node = np.zeros(shape=[1, 0], dtype=np.float32)
                    track_meta_data, cluster_meta_data = self.get_meta(event_data, event_ind, c_inds)

                    graph = {'nodes': nodes,
                             'globals': global_node,
                             'senders': np.array(senders, dtype=np.int64),
                             'receivers': np.array(receivers, dtype=np.int64),
                             'edges': edges}

                    target = truth_particle_E.astype(np.float32)

                    preprocessed_data.append((graph, target, track_meta_data, cluster_meta_data))

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
        batch_track_meta = []
        batch_cluster_meta = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                batch_track_meta.append(file_data[i][2])
                batch_cluster_meta.append(file_data[i][3])

                if len(batch_graphs) == self.batch_size:
                    batch_targets = np.array(batch_targets).astype(np.float32)
                    batch_queue.put((batch_graphs, batch_targets, batch_track_meta, batch_cluster_meta))

                    batch_graphs = []
                    batch_targets = []
                    batch_track_meta = []
                    batch_cluster_meta = []

            file_num += self.num_procs

        if len(batch_graphs) > 0:
            batch_targets = np.array(batch_targets).astype(np.float32)
            batch_queue.put((batch_graphs, batch_targets, batch_track_meta, batch_cluster_meta))


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