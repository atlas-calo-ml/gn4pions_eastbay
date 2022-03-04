import numpy as np
import glob
import os
import uproot as ur
import time
from multiprocessing import Process, Queue, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
import random

class MPGraphDataGenerator:
    """DataGenerator class for extracting and formating data from list of root files"""
    def __init__(self,
                 pi0_file_list: list,
                 pion_file_list: list,
                 cellGeo_file: str,
                 batch_size: int,
                 cluster_E_cut: float = None,
                 shuffle: bool = True,
                 num_procs: int = 32,
                 preprocess: bool = False,
                 output_dir: str = None):
        """Initialization"""

        self.preprocess = preprocess
        self.output_dir = output_dir
        self.cluster_E_cut = cluster_E_cut

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
        self.meta_features = ['file_name', 'event_ind', 'cluster_ind', 'cluster_E', 'cluster_Pt',
                              'cluster_Eta', 'cluster_Phi', 'cluster_nCells', 'cluster_ENG_CALIB_TOT', 'type']

        self.cellGeo_data = self.cellGeo_data.arrays(library='np')
        self.cellGeo_ID = self.cellGeo_data['cell_geo_ID'][0]
        self.sorter = np.argsort(self.cellGeo_ID)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if self.shuffle: np.random.shuffle(self.file_list)
        
        self.num_procs = num_procs
        self.procs = []

        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

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
        global_node = np.log10(event_data['cluster_E'][event_ind][cluster_ind])
        
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
        
        return nodes, np.array([global_node]), cluster_num_nodes, cell_IDmap
    
    def get_edges(self, cluster_num_nodes, cell_IDmap):
        """ 
        Reading edge features 
        Resturns senders, receivers, and edges    
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

    def get_meta(self, event_data, event_ind, cluster_ind):
        """ 
        Reading meta data
        Returns senders, receivers, and edges    
        """ 
        meta_data = [] 
        meta_data.append(event_ind)
        meta_data.append(cluster_ind)
        for f in self.meta_features[3:-1]:
            meta_data.append(event_data[f][event_ind][cluster_ind])
        
        return meta_data

    def preprocessor(self, worker_id):
        file_num = worker_id
        while file_num < self.num_files:
            print(f"Proceesing file {os.path.basename(self.pion_file_list[file_num])}")
            f_name = self.pion_file_list[file_num]
            event_tree = ur.open(f_name)['EventTree']
            num_events = event_tree.num_entries

            event_data = event_tree.arrays(library='np')

            preprocessed_data = []

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]
                
                for i in range(num_clusters):
                    cluster_calib_E = self.get_cluster_calib(event_data, event_ind, i)
                    
                    if cluster_calib_E is None:
                        continue
                        
                    nodes, global_node, cluster_num_nodes, cell_IDmap = self.get_nodes(event_data, event_ind, i)
                    if self.cluster_E_cut and 10**global_node[0]<self.cluster_E_cut:
                        continue
                    senders, receivers, edges = self.get_edges(cluster_num_nodes, cell_IDmap)

                    graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                        'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                        'edges': edges.astype(np.float32)}
                    target = np.reshape([cluster_calib_E.astype(np.float32), 1], [1,2])
                    
                    meta_data = [f_name]
                    meta_data.extend(self.get_meta(event_data, event_ind, i))
                    meta_data.append('pion')

                    preprocessed_data.append((graph, target, meta_data))

            print(f"Finished file {os.path.basename(self.pion_file_list[file_num])}")
            print(f"Proceesing file {os.path.basename(self.pi0_file_list[file_num])}")
            f_name = self.pi0_file_list[file_num]
            event_tree = ur.open(f_name)['EventTree']
            num_events = event_tree.num_entries

            event_data = event_tree.arrays(library='np')

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]
                
                for i in range(num_clusters):
                    cluster_calib_E = self.get_cluster_calib(event_data, event_ind, i)
                    
                    if cluster_calib_E is None:
                        continue
                        
                    nodes, global_node, cluster_num_nodes, cell_IDmap = self.get_nodes(event_data, event_ind, i)
                    if self.cluster_E_cut and 10**global_node[0]<self.cluster_E_cut:
                        continue
                    senders, receivers, edges = self.get_edges(cluster_num_nodes, cell_IDmap)
                    
                    graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                        'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                        'edges': edges.astype(np.float32)}
                    target = np.reshape([cluster_calib_E.astype(np.float32), 0], [1,2])

                    meta_data = [f_name]
                    meta_data.extend(self.get_meta(event_data, event_ind, i))
                    meta_data.append('pi0')

                    preprocessed_data.append((graph, target, meta_data))

            print(f"Finished file {os.path.basename(self.pi0_file_list[file_num])}")
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
        batch_meta = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                batch_meta.append(file_data[i][2])
                    
                if len(batch_graphs) == self.batch_size:
                    batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
                    
                    batch_queue.put((batch_graphs, batch_targets, batch_meta))
                    
                    batch_graphs = []
                    batch_targets = []
                    batch_meta = []

            file_num += self.num_procs
                    
        if len(batch_graphs) > 0:
            batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
            
            batch_queue.put((batch_graphs, batch_targets, batch_meta))

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
        # for file in self.file_list:
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
            
        
if __name__ == '__main__':
    data_dir = '/usr/workspace/pierfied/preprocessed/data/'
    out_dir = '/usr/workspace/pierfied/preprocessed/preprocessed_data/'
    pion_files = np.sort(glob.glob(data_dir+'user*.root'))
    
    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                  cellGeo_file=data_dir+'cell_geo.root',
                                  batch_size=32,
                                  shuffle=False,
                                  num_procs=32,
                                  preprocess=True,
                                  output_dir=out_dir)

    gen = data_gen.generator()
    
    from tqdm.auto import tqdm
     
    for batch in tqdm(gen):
        pass
        
    exit()
