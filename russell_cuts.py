'''
ML4P Point Cloud Data Script
Author: Russell Bate
russell.bate@cern.ch
russellbate@phas.ubc.ca

Notes: Version 2 of the STMC data script.
- single tracks
- clusters within DeltaR of 1.2 of track
- energy weighted cluster average '''
#====================
# Load Utils ========
#====================

import numpy as np
import uproot as ur
import awkward as ak
import time as t
import os
import argparse
from copy import deepcopy
from glob import glob
from tqdm import tqdm

Nfile=1

print()
print('='*43)
print('== Single Track Multiple Cluster Script ==')
print('='*43)
print()
print("Awkward version: "+str(ak.__version__))
print("Uproot version: "+str(ur.__version__))


## Read in Parameters
#=============================================================================
parser = argparse.ArgumentParser(description='Inputs for STMC track script.')

parser.add_argument('--nFile', action="store", dest='nf', default=1,
                   type=int)

args = parser.parse_args()

Nfile = args.nf

print('Working on {} files'.format(Nfile))

#====================
# Functions =========
#====================

def DeltaR(coords, ref):
    ''' Straight forward function, expects Nx2 inputs for coords, 1x2 input for ref '''
    ref = np.tile(ref, (len(coords[:,0]), 1))
    DeltaCoords = np.subtract(coords, ref)
    ## Mirroring ##
    gt_pi_mask = DeltaCoords > np.pi
    lt_pi_mask = DeltaCoords < - np.pi
    DeltaCoords[lt_pi_mask] = DeltaCoords[lt_pi_mask] + 2*np.pi
    DeltaCoords[gt_pi_mask] = DeltaCoords[gt_pi_mask] - 2*np.pi
    return np.sqrt(DeltaCoords[:,0]**2 + DeltaCoords[:,1]**2)

def find_max_dim_tuple(events, event_dict):
    nEvents = len(events)
    max_clust = 0
    
    for i in range(nEvents):
        event = events[i,0]
        track_nums = events[i,1]
        clust_nums = events[i,2]
        
        clust_num_total = 0
        # set this to six for now to handle single track events, change later
        track_num_total = 10 # max 9 but keep a buffer of 1
        
        # Check if there are clusters, None type object may be associated with it
        if clust_nums is not None:
            # Search through cluster indices
            for clst_idx in clust_nums:
                nInClust = len(event_dict['cluster_cell_ID'][event][clst_idx])
                # add the number in each cluster to the total
                clust_num_total += nInClust

        total_size = clust_num_total + track_num_total
        if total_size > max_clust:
            max_clust = total_size
    
    # 6 for energy, eta, phi, rperp, track flag, sample layer
    return (nEvents, max_clust, 6)

def dict_from_tree(tree, branches=None, np_branches=None):
    ''' Loads branches as default awkward arrays and np_branches as numpy arrays. '''
    dictionary = dict()
    if branches is not None:
        for key in branches:
            branch = tree.arrays()[key]
            dictionary[key] = branch
            
    if np_branches is not None:
        for np_key in np_branches:
            np_branch = np.ndarray.flatten(tree.arrays()[np_key].to_numpy())
            dictionary[np_key] = np_branch
    
    if branches is None and np_branches is None:
        raise ValueError("No branches passed to function.")
        
    return dictionary

def find_index_1D(values, dictionary):
    ''' Use a for loop and a dictionary. values are the IDs to search for. dict must be in format 
    (cell IDs: index) '''
    idx_vec = np.zeros(len(values), dtype=np.int32)
    for i in range(len(values)):
        idx_vec[i] = dictionary[values[i]]
    return idx_vec

#====================
# Metadata ==========
#====================
event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", 'cluster_nCells', "nCluster", "eventNumber",
                  "nTrack", "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", 'trackPt', 'trackP',
                  'trackMass', 'trackEta', 'trackPhi', 'truthPartE', 'cluster_ENG_CALIB_TOT', "cluster_E", 'truthPartPt']

ak_event_branches = ["cluster_nCells", "cluster_cell_ID", "cluster_cell_E", "cluster_nCells",
                  "nTruthPart", "truthPartPdgId", "cluster_Eta", "cluster_Phi", "trackPt", "trackP",
                  "trackMass", "trackEta", "trackPhi", "truthPartE", "cluster_ENG_CALIB_TOT", "cluster_E", "truthPartPt"]

np_event_branches = ["nCluster", "eventNumber", "nTrack", "nTruthPart"]

geo_branches = ["cell_geo_ID", "cell_geo_eta", "cell_geo_phi", "cell_geo_rPerp", "cell_geo_sampling"]


#======================================
# Track related meta-data
#======================================
trk_em_eta = ['trackEta_EMB2', 'trackEta_EME2']
trk_em_phi = ['trackPhi_EMB2', 'trackPhi_EME2']

trk_proj_eta = ['trackEta_EMB1', 'trackEta_EMB2', 'trackEta_EMB3',
    'trackEta_EME1', 'trackEta_EME2', 'trackEta_EME3', 'trackEta_HEC0',
    'trackEta_HEC1', 'trackEta_HEC2', 'trackEta_HEC3', 'trackEta_TileBar0',
    'trackEta_TileBar1', 'trackEta_TileBar2', 'trackEta_TileGap1',
    'trackEta_TileGap2', 'trackEta_TileGap3', 'trackEta_TileExt0',
    'trackEta_TileExt1', 'trackEta_TileExt2']
trk_proj_phi = ['trackPhi_EMB1', 'trackPhi_EMB2', 'trackPhi_EMB3',
    'trackPhi_EME1', 'trackPhi_EME2', 'trackPhi_EME3', 'trackPhi_HEC0',
    'trackPhi_HEC1', 'trackPhi_HEC2', 'trackPhi_HEC3', 'trackPhi_TileBar0',
    'trackPhi_TileBar1', 'trackPhi_TileBar2', 'trackPhi_TileGap1',
    'trackPhi_TileGap2', 'trackPhi_TileGap3', 'trackPhi_TileExt0',
    'trackPhi_TileExt1', 'trackPhi_TileExt2']
calo_numbers = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
eta_trk_dict = dict(zip(trk_proj_eta, calo_numbers))

calo_layers = ['EMB1', 'EMB2', 'EMB3', 'EME1', 'EME2', 'EME3', 'HEC0', 'HEC1',
    'HEC2', 'HEC3', 'TileBar0', 'TileBar1', 'TileBar2', 'TileGap1', 'TileGap2',
    'TileGap3', 'TileExt0', 'TileExt1', 'TileExt2']
calo_numbers2 = [1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
calo_dict = dict(zip(calo_numbers2, calo_layers))

fixed_z_numbers = [5,6,7,8,9,10,11]
fixed_z_vals = [3790.03, 3983.68, 4195.84, 4461.25, 4869.50, 5424.50, 5905.00]
z_calo_dict = dict(zip(fixed_z_numbers, fixed_z_vals))

fixed_r_numbers = [1,2,3,12,13,14,15,16,17,18,19,20]
fixed_r_vals = [1532.18, 1723.89, 1923.02, 2450.00, 2995.00, 3630.00, 3215.00,
                3630.00, 2246.50, 2450.00, 2870.00, 3480.00]
r_calo_dict = dict(zip(fixed_r_numbers, fixed_r_vals))


#====================
# File setup ========
#====================

pion_dir = '/clusterfs/ml4hep/mpettee/ml4pions/data/user.angerami.mc16_13TeV.900247.PG_singlepion_logE0p2to2000.e8312_e7400_s3170_r12383.v01-45-gaa27bcb_OutputStream/'
fileNames = sorted(glob(pion_dir+"*.root"))[:Nfile]

#====================
# Load Data Files ===
#====================

## GEOMETRY DICTIONARY ##
geo_file = ur.open('/clusterfs/ml4hep/mpettee/ml4pions/data/cell_geo.root')
CellGeo_tree = geo_file["CellGeo"]
geo_dict = dict_from_tree(tree=CellGeo_tree, branches=None, np_branches=geo_branches)

# cell geometry data
cell_geo_ID = geo_dict['cell_geo_ID']
cell_ID_dict = dict(zip(cell_geo_ID, np.arange(len(cell_geo_ID))))

## MEMORY MAPPED ARRAY ALLOCATION ##
X_large = np.lib.format.open_memmap('/clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/X_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,1500,6), fortran_order=False, version=None)
Y_large = np.lib.format.open_memmap('/clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/Y_large.npy', mode='w+', dtype=np.float64,
                       shape=(2500000,3), fortran_order=False, version=None)
Eta_large = np.empty(2500000)


# Pre-Loop Definitions ##
#======================================
k = 1 # tally used to keep track of file number
tot_nEvts = 0 # used for keeping track of total number of events
max_nPoints = 0 # used for keeping track of the largest 'point cloud'
t_tot = 0 # total time
# for event dictionary
events_prefix = ''
num_zero_tracks = 0


## Main File Loop ##
#======================================
for currFile in fileNames:
    
    # Check for file, a few are missing
    if not os.path.isfile(events_prefix+currFile):
        print()
        print('File '+events_prefix+currFile+' not found..')
        print()
        k += 1
        continue
    
    else:
        print()
        print('Working on File: '+str(currFile)+' - '+str(k)+'/'+str(Nfile))
        k += 1
        
    t0 = t.time()
    ## EVENT DICTIONARY ##
    event = ur.open(events_prefix+currFile)
    event_tree = event["EventTree"]
    event_dict = dict_from_tree(tree=event_tree, branches=ak_event_branches, np_branches=np_event_branches)
    
    ## TRACK DICTIONARY ##
    track_dict = dict_from_tree(tree=event_tree,
                branches=deepcopy(trk_proj_eta+trk_proj_phi))
    
    #===================
    # APPLY CUTS =======
    #===================
    # create ordered list of events to use for index slicing
    nEvents = len(event_dict['eventNumber'])
    all_events = np.arange(0,nEvents,1,dtype=np.int32)

    # SINGLE TRACK CUT
    single_track_mask = event_dict['nTrack'] == np.full(nEvents, 1)
    single_track_filter = all_events[single_track_mask]
    
    # TRACKS WITH CLUSTERS
    nCluster = event_dict['nCluster'][single_track_filter]
    nz_clust_mask = nCluster != 0
    filtered_event = single_track_filter[nz_clust_mask]
    t1 = t.time()
    events_cuts_time = t1 - t0
    
    #============================================#
    ## CREATE INDEX ARRAY FOR TRACKS + CLUSTERS ##
    #============================================#
    event_indices = []
    t0 = t.time()

    for evt in filtered_event:

        # pull cluster number, don't need zero as it's loaded as a np array
        nClust = event_dict["nCluster"][evt]
        cluster_idx = np.arange(nClust)

        # Notes: this will need to handle more complex scenarios in the future for tracks with
        # no clusters

        ## DELTA R ##
        # pull coordinates of tracks and clusters from event
        # we can get away with the zeroth index because we are working with single track events
        trackCoords = np.array([event_dict["trackEta"][evt][0],
                                 event_dict["trackPhi"][evt][0]])
        clusterCoords = np.stack((event_dict["cluster_Eta"][evt].to_numpy(),
                                   event_dict["cluster_Phi"][evt].to_numpy()), axis=1)

        _DeltaR = DeltaR(clusterCoords, trackCoords)
        DeltaR_mask = _DeltaR < 1.2
        matched_clusters = cluster_idx[DeltaR_mask]

        ## CREATE LIST ##
        # Note: currently do not have track only events. Do this in the future    
        if np.count_nonzero(DeltaR_mask) > 0:
            event_indices.append((evt, 0, matched_clusters))
    
    event_indices = np.array(event_indices, dtype=np.object_)
    t1 = t.time()
    indices_time = t1 - t0
    
    #=========================#
    ## DIMENSIONS OF X ARRAY ##
    #=========================#
    t0 = t.time()
    max_dims = find_max_dim_tuple(event_indices, event_dict)
    evt_tot = max_dims[0]
    tot_nEvts += max_dims[0]
    # keep track of the largest point cloud to use for saving later
    if max_dims[1] > max_nPoints:
        max_nPoints = max_dims[1]
    
    # Create arrays
    Y_new = np.zeros((max_dims[0],3))
    X_new = np.zeros(max_dims)
    Eta_new = np.zeros(max_dims[0])
    t1 = t.time()
    find_create_max_dims_time = t1 - t0    
    
    #===================#
    ## FILL IN ENTRIES ##==============================================================
    #===================#
    t0 = t.time()
    for i in tqdm(range(max_dims[0]), desc="Filling entries"):
        # pull all relevant indices
        evt = event_indices[i,0]
        track_idx = event_indices[i,1]
        # recall this now returns an array
        cluster_nums = event_indices[i,2]

        ## Centering ##
        trk_bool_em = np.zeros(2, dtype=bool)
        trk_full_em = np.empty((2,2))
    
        for l, (eta_key, phi_key) in enumerate(zip(trk_em_eta, trk_em_phi)):

            eta_em = track_dict[eta_key][evt][track_idx]
            phi_em = track_dict[phi_key][evt][track_idx]

            if np.abs(eta_em) < 2.5 and np.abs(phi_em) <= np.pi:
                trk_bool_em[l] = True
                trk_full_em[l,0] = eta_em
                trk_full_em[l,1] = phi_em
                
        nProj_em = np.count_nonzero(trk_bool_em)
        if nProj_em == 1:
            eta_ctr = trk_full_em[trk_bool_em, 0]
            phi_ctr = trk_full_em[trk_bool_em, 1]
            
        elif nProj_em == 2:
            trk_av_em = np.mean(trk_full_em, axis=1)
            eta_ctr = trk_av_em[0]
            phi_ctr = trk_av_em[1]
            
        elif nProj_em == 0:
            eta_ctr = event_dict['trackEta'][evt][track_idx]
            phi_ctr = event_dict['trackPhi'][evt][track_idx]      
        
        ##############
        ## CLUSTERS ##
        ##############
        # set up to have no clusters, further this with setting up the same thing for tracks
        target_ENG_CALIB_TOT = -1
        if cluster_nums is not None:

            # find averaged center of clusters
            cluster_Eta = event_dict['cluster_Eta'][evt].to_numpy()
            cluster_Phi = event_dict['cluster_Phi'][evt].to_numpy()
            cluster_E = event_dict['cluster_E'][evt].to_numpy()
            cl_E_tot = np.sum(cluster_E)

            nClust_current_total = 0
            target_ENG_CALIB_TOT = 0
            for c in cluster_nums:            
                # cluster data
                target_ENG_CALIB_TOT += event_dict['cluster_ENG_CALIB_TOT'][evt][c]
                cluster_cell_ID = event_dict['cluster_cell_ID'][evt][c].to_numpy()
                nInClust = len(cluster_cell_ID)
                cluster_cell_E = event_dict['cluster_cell_E'][evt][c].to_numpy()            
                cell_indices = find_index_1D(cluster_cell_ID, cell_ID_dict)

                cluster_cell_Eta = geo_dict['cell_geo_eta'][cell_indices]
                cluster_cell_Phi = geo_dict['cell_geo_phi'][cell_indices]
                cluster_cell_rPerp = geo_dict['cell_geo_rPerp'][cell_indices]
                cluster_cell_sampling = geo_dict['cell_geo_sampling'][cell_indices]

                # input all the data
                # note here we leave the fourth entry zeros (zero for flag!!!)
                low = nClust_current_total
                high = low + nInClust
                X_new[i,low:high,0] = cluster_cell_E
                # Normalize to average cluster centers
                X_new[i,low:high,1] = cluster_cell_Eta - eta_ctr
                X_new[i,low:high,2] = cluster_cell_Phi - eta_ctr
                X_new[i,low:high,3] = cluster_cell_rPerp
                X_new[i,low:high,5] = cluster_cell_sampling

                nClust_current_total += nInClust

        #####################
        ## TARGET ENERGIES ##
        #####################
        # this should be flattened or loaded as np array instead of zeroth index in future
        Y_new[i,0] = event_dict['truthPartE'][evt][0]
        Y_new[i,1] = event_dict['truthPartPt'][evt][track_idx]
        Y_new[i,2] = target_ENG_CALIB_TOT
        
        #########
        ## ETA ##
        #########
        # again only get away with this because we have a single track
        Eta_new[i] = event_dict["trackEta"][evt][track_idx]

        ############
        ## TRACKS ##
        ############
        
        trk_bool = np.zeros(len(calo_numbers), dtype=bool)
        trk_full = np.empty((len(calo_numbers), 4))
        
        for j, (eta_key, phi_key) in enumerate(zip(trk_proj_eta, trk_proj_phi)):
            
            cnum = eta_trk_dict[eta_key]
            layer = calo_dict[cnum]
            
            eta = track_dict[eta_key][evt][track_idx]
            phi = track_dict[phi_key][evt][track_idx]
            
            if np.abs(eta) < 2.5 and np.abs(phi) <= np.pi:
                trk_bool[j] = True
                trk_full[j,0] = eta
                trk_full[j,1] = phi
                trk_full[j,3] = cnum
                
                if cnum in fixed_r_numbers:
                    rPerp = r_calo_dict[cnum]
                    
                elif cnum in fixed_z_numbers:
                    z = z_calo_dict[cnum]
                    aeta = np.abs(eta)
                    rPerp = z*2*np.exp(aeta)/(np.exp(2*aeta) - 1)
                    
                else:
                    raise ValueError('Calo sample num not found in dicts..')
                
                if rPerp < 0:
                    print()
                    print('Found negative rPerp'); print()
                    print('Event number: {}'.format(evt))
                    print('Eta: {}'.format(eta))
                    print('Phi: {}'.format(phi))
                    print('rPerp: {}'.format(rPerp))
                    raise ValueError('Found negative rPerp')
                    
                trk_full[j,2] = rPerp
                
        # Fill in track array
        trk_proj_num = np.count_nonzero(trk_bool)
        
        if trk_proj_num == 0:
            trk_proj_num = 1
            trk_arr = np.empty((1, 6))
            num_zero_tracks += 1
            trk_arr[:,0] = event_dict['trackP'][evt][track_idx]
            trk_arr[:,1] = event_dict['trackEta'][evt][track_idx] - eta_ctr
            trk_arr[:,2] = event_dict['trackPhi'][evt][track_idx] - phi_ctr
            trk_arr[:,3] = 1532.18 # just place it in EMB1
            trk_arr[:,4] = 1 # track flag
            trk_arr[:,5] = 1 # place layer in EMB1
        else:
            trk_arr = np.empty((trk_proj_num, 6))
            trackP = event_dict['trackP'][evt][track_idx]
            trk_arr[:,1:4] = np.ndarray.copy(trk_full[trk_bool,:3])
            trk_arr[:,4] = np.ones(trk_proj_num)
            trk_arr[:,5] = np.ndarray.copy(trk_full[trk_bool,3])
            trk_arr[:,0] = trackP/trk_proj_num

            trk_arr[:,1] = trk_arr[:,1] - eta_ctr
            trk_arr[:,2] = trk_arr[:,2] - phi_ctr

        X_new[i,high:high+trk_proj_num,:] = np.ndarray.copy(trk_arr)
    
    #=========================================================================#
    t1 = t.time()
    array_construction_time = t1 - t0
    
    #=======================#
    ## ARRAY CONCATENATION ##
    #=======================#
    t0 = t.time()
    # Write to X
    old_tot = tot_nEvts - max_dims[0]
    X_large[old_tot:tot_nEvts, :max_dims[1], :6] = np.ndarray.copy(X_new)
    # pad the remainder with zeros (just to be sure)
    fill_shape = (tot_nEvts - old_tot, 1500 - max_dims[1], 6)
    X_large[old_tot:tot_nEvts, max_dims[1]:1500, :6] = np.zeros(fill_shape)
    
    # Write to Y
    Y_large[old_tot:tot_nEvts,:] = np.ndarray.copy(Y_new)
    
    # Eta
    Eta_large[old_tot:tot_nEvts] = np.ndarray.copy(Eta_new)
        
    t1 = t.time()
    time_to_memmap = t1-t0
    thisfile_t_tot = events_cuts_time+find_create_max_dims_time+indices_time\
          +array_construction_time+time_to_memmap
    t_tot += thisfile_t_tot
    
    print('Array dimension: '+str(max_dims))
    print('Number of null track projection: {:.1f} seconds'.format(num_zero_tracks))
    print('Time to create dicts and select events: {:.1f} seconds'.format(events_cuts_time))
    print('Time to find dimensions and make new array: {:.1f} seconds'.format(find_create_max_dims_time))
    print('Time to construct index array: {:.1f} seconds'.format(indices_time))
    print('Time to populate elements: {:.1f} seconds'.format(array_construction_time))
    print('Time to copy to memory map: {:.1f} seconds'.format(time_to_memmap))
    print('Time for this file: {:.1f} seconds'.format(thisfile_t_tot))
    print('Total events: '+str(tot_nEvts))
    print('Current size: '+str((tot_nEvts,max_nPoints,6)))
    print('Total time: {:.1f} seconds'.format(t_tot))
    print()

t0 = t.time()
X = np.lib.format.open_memmap('/clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/X_STMC_v2_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, max_nPoints, 6))
np.copyto(dst=X, src=X_large[:tot_nEvts,:max_nPoints,:], casting='same_kind', where=True)
del X_large
os.system('rm /clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/X_large.npy')

Y = np.lib.format.open_memmap('/clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/Y_STMC_v2_'+str(Nfile)+'_files.npy',
                             mode='w+', dtype=np.float64, shape=(tot_nEvts, 3))
np.copyto(dst=Y, src=Y_large[:tot_nEvts,:], casting='same_kind', where=True)
del Y_large
os.system('rm /clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/Y_large.npy')

np.save('/clusterfs/ml4hep/mpettee/ml4pions/data/onetrack_multicluster/pion_files_russell/Eta_STMC_v2_'+str(Nfile)+'_files', Eta_large[:tot_nEvts])


t1 = t.time()
print()
print('Time to copy new and delete old: {:.1f} seconds'.format(t1-t0))
print()

