import pandas as pd
import numpy as np
import glob
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import networkx as nx # for visualizing graphs
from sklearn.preprocessing import StandardScaler

import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch.optim as optim
from copy import deepcopy
from torch_geometric.nn import GINConv
from torch.utils.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.metrics import *
from torch.nn import Sequential, Linear, ReLU
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

def clean_dataframe(df, max_n_cols=35): 
    ### Start the dataframe of inputs 
    # df2 = pd.DataFrame(pd.DataFrame(df.cluster_E.to_list())[0]) # just take the leading cluster E 
    df2 = pd.DataFrame(pd.DataFrame(df.cluster_E.to_list(), columns=["cluster_e_"+str(x) for x in np.arange(max_n_cols)]))
    
    ### Add track pT & truth particle E 
    track_pt = np.array(df.trackPt.explode())
    truth_particle_e = np.array(df.truthPartE.explode())
    track_eta = np.array(df.trackEta.explode())
    track_phi = np.array(df.trackPhi.explode())
    track_z0 = np.array(df.trackZ0.explode())

    df2["track_pt"] = track_pt
    df2["track_eta"] = track_eta
    df2["track_phi"] = track_phi
    df2["track_z0"] = track_z0
    df2["truth_particle_e"] = truth_particle_e
        
    ### Cluster_E > 0.5
    df2 = df2[df2.cluster_e_0 > 0.5]

    ### Lose outliers in track pT 
    df2 = df2[df2.track_pt < 5000]

    ### Cast as float
    df2 = df2.astype('float32')

    ### Add the log of all energy variables
    for var in df2.keys():
        if var in ["track_eta", "track_phi", "track_z0"]:
            continue
        else:
            df2['log10_'+var] = np.log10(df2[var])

    ### Drop infs/NaNs 
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df2 = df2.fillna(0)
    
    ### Reduce variables
    vars = [
    'log10_cluster_e_0', 
    'log10_track_pt',
    'track_eta', 
    'track_phi',
    'track_z0',
    'log10_truth_particle_e'
             ]
    
    for i in np.arange(1,max_n_cols):
        vars += ['log10_cluster_e_'+str(i)]
    
    df2 = df2[vars]
    
    ### Standardize inputs
    sc = StandardScaler()
    x = df2.values
    x = sc.fit_transform(x)
    
    for i in range(len(vars)):
        df2[vars[i]+"_scaled"] = x[:,i]
        
    ### Get rid of any duplicate target values
    df2.reset_index(drop=True, inplace=True)
    series = df2.log10_truth_particle_e_scaled
    indices_to_drop = series[series.duplicated()].index
    df2.drop(indices_to_drop, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    
    return df2

class PionDataset_Regress(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, dataframe, cluster_features, track_features):
        'Initialization'
        self.dataframe = dataframe
        self.cluster_features = cluster_features
        self.track_features = track_features
        print("")

    def __len__(self):
        'Denotes the total number of samples'
        dataframe = self.dataframe
        return len(dataframe)

    def __getitem__(self, index):
        'Generates one sample of data'  
        dataframe = self.dataframe
        
        ### define nodes 
        cluster_node = np.array(dataframe.iloc[index][cluster_features])
        cluster_node = np.concatenate([cluster_node, np.zeros(len(track_features))]) # cluster features come first
        
        track_node = np.array(dataframe.iloc[index][track_features])
        track_node = np.concatenate([np.zeros(len(cluster_features)), track_node]) # cluster features come first

        # shape = (num_nodes, num_node_features) = (2, 3)
        nodes = np.vstack([cluster_node, track_node]) 
        
        ### define edges (fully-connected, but no self-loops)
        edges = [(i,j) for i in range(nodes.shape[0]) for j in range(nodes.shape[0]) if i != j]
        edges_reversed = [(i,j) for i in range(nodes.shape[0]) for j in range(nodes.shape[0]) if i != j]
        edge_index = np.row_stack([edges,edges_reversed])
        
        ### Define target 
        target = np.array([dataframe.iloc[index]['log10_truth_particle_e']])
        
        # Convert to torch objects
        nodes = torch.Tensor(nodes)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        target = torch.tensor(target)
#         edge_attr = torch.Tensor(is_skeleton_edge)
        
        return Data(x=nodes, y=target, edge_index=edge_index.t().contiguous(), 
#                     edge_attr=edge_attr
                   )
class GIN_Regress(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(GIN_Regress, self).__init__()
        self.num_layers = args["num_layers"]

        self.pre_mp = nn.Linear(input_size, hidden_size)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for l in range(self.num_layers):
            layer = Sequential(
                Linear(hidden_size, hidden_size), 
                nn.ReLU(), 
                Linear(hidden_size, hidden_size)
            )
            self.convs.append(GINConv(layer))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.post_mp = Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch
        x = self.pre_mp(x)
        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
        x = self.convs[-1](x, edge_index)
        x = pyg_nn.global_mean_pool(x, batch)
        x = self.post_mp(x)
        return x

    def loss(self, pred, target):
        pred = pred.reshape(pred.shape[0])
        return F.mse_loss(pred, target)
    
def train(train_loader, val_loader, test_loader, args, num_node_features, output_size, device="cpu"):
    model = GIN_Regress(num_node_features, args["hidden_size"], 1, args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)
    best_model = None
    max_val = -1
    for epoch in range(args["epochs"]):
        t0 = time.time() 
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.graph_label
            loss = model.loss(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        if val_acc > max_val:
            max_val = val_acc
            best_model = deepcopy(model)
        test_acc = test(test_loader, model, device)
        log = "Epoch {} (took {:.2f} seconds): Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}"
        print(log.format(epoch + 1, time.time() - t0, train_acc, val_acc, test_acc, total_loss))
    return best_model

def test(loader, model, device='cuda'):
    model.eval()
    ratio = []
    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)[:,0] # to make the shapes match
            target = batch.graph_label
            ratio.append(torch.median(torch.divide(pred, target))) 
    return np.mean(np.array(ratio))

def test_after_training(loader, model, device='cuda'):
    model.eval()
    preds = []
    targets = []
    ratio = []
    batch_num = 0
    for batch in loader:
#         print("====== BATCH {} ======".format(batch_num))
        batch.to(device)
        with torch.no_grad():
            pred = model(batch)[:,0] # to make the shapes match
            preds.append(pred)
#             print("Predicted energies:", pred)
            target = batch.graph_label
            targets.append(target)
#             print("Truth energies:", target)
#             print("Ratios:", torch.divide(pred, target))
        ratio.append(torch.median(torch.divide(pred, target))) 
        batch_num += 1
#         print("Median ratio per batch: {}".format(torch.median(torch.divide(pred, target))))
    preds_reshaped = torch.cat([torch.stack(preds[:-1]).view(-1, 1), preds[-1].view(-1, 1)])
    targets_reshaped = torch.cat([torch.stack(targets[:-1]).view(-1, 1), targets[-1].view(-1, 1)])
    return np.median(np.array(ratio)), preds_reshaped, targets_reshaped

### Load data (multiple files)
n_files = 10
max_n_cols = 33 # 33 for 10 files, 35 for 100
pion_files = glob.glob("../data/onetrack_multicluster/pion_files/*.npy")
df_pion = pd.concat([pd.DataFrame(np.load(file, allow_pickle=True).item()) for file in tqdm(pion_files[:n_files], desc="Loading .npy files")])
print("Pion dataframe has {:,} events.".format(df_pion.shape[0]))    
    
df = clean_dataframe(df_pion, max_n_cols=max_n_cols) 

cluster_features = ['log10_cluster_e_0_scaled', 'log10_cluster_e_1_scaled', 'log10_cluster_e_2_scaled',
#        'log10_cluster_e_3_scaled', 'log10_cluster_e_4_scaled',
#        'log10_cluster_e_5_scaled', 'log10_cluster_e_6_scaled',
#        'log10_cluster_e_7_scaled', 'log10_cluster_e_8_scaled',
#        'log10_cluster_e_9_scaled', 'log10_cluster_e_10_scaled',
#        'log10_cluster_e_11_scaled', 'log10_cluster_e_12_scaled',
#        'log10_cluster_e_13_scaled', 'log10_cluster_e_14_scaled',
#        'log10_cluster_e_15_scaled', 'log10_cluster_e_16_scaled',
#        'log10_cluster_e_17_scaled', 'log10_cluster_e_18_scaled',
#        'log10_cluster_e_19_scaled', 'log10_cluster_e_20_scaled',
#        'log10_cluster_e_21_scaled', 'log10_cluster_e_22_scaled',
#        'log10_cluster_e_23_scaled', 'log10_cluster_e_24_scaled',
#        'log10_cluster_e_25_scaled', 'log10_cluster_e_26_scaled',
#        'log10_cluster_e_27_scaled', 'log10_cluster_e_28_scaled',
#        'log10_cluster_e_29_scaled', 'log10_cluster_e_30_scaled',
#        'log10_cluster_e_31_scaled', 'log10_cluster_e_32_scaled',
#        'log10_cluster_e_33_scaled', 'log10_cluster_e_34_scaled',
]
track_features = ['log10_track_pt_scaled', 'track_eta_scaled', 'track_phi_scaled', 'track_z0_scaled']

print("Converting to graphs...")
t0 = time.time()
pions = PionDataset_Regress(df, cluster_features, track_features)
graphs = GraphDataset.pyg_to_graphs(pions) 
dataset = GraphDataset(graphs, task="graph", minimum_node_per_graph=0)
print("Took {:.2f} minutes.".format((time.time() - t0)/60))
print("Dataset has {:,} and {} node features".format(len(dataset), dataset.num_node_features))

print("\n========= EXAMPLE GRAPH =========")
graph = dataset[16]
print(graph)
print(graph.node_feature)
print(graph.graph_label)
print("===================================\n")


args = {
    "device" : 'cpu', #'cuda' if torch.cuda.is_available() else 'cpu',
    "hidden_size" : 10,
    "epochs" : 10,
    "lr" : 0.001,
    "num_layers": 3,
    "batch_size": 32,
}

dataset_train, dataset_val, dataset_test = \
    dataset.split(transductive=False, split_ratio = [0.8, 0.1, 0.1])

num_node_features = len(cluster_features)+len(track_features)

train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(),\
    batch_size=args["batch_size"], shuffle=True)
val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(),\
    batch_size=args["batch_size"])
test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(),\
    batch_size=args["batch_size"])

print("Training...")
best_model = train(train_loader, val_loader, test_loader, args, 
                   num_node_features, 1, args["device"])

train_acc = test(train_loader, best_model, args["device"])
val_acc = test(val_loader, best_model, args["device"])
test_acc = test(test_loader, best_model, args["device"])

# Values printed are best median ratio of predicted energy to target:
log = "Best model: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}"
print(log.format(train_acc, val_acc, test_acc))

median_ratio, preds, targets = test_after_training(test_loader, best_model, args["device"])
print("Median ratio from best model = {:.2f}".format(median_ratio))
      
print("Evaluating on test set...")
test_df = df.loc[df['log10_truth_particle_e'].isin(np.array(targets[:,0]))]
test_df = test_df.sort_values('log10_truth_particle_e')

nn_outputs = pd.DataFrame(np.vstack([np.array(preds[:,0]), np.array(targets[:,0])]).T,
             columns = ["preds", "targets"])
nn_outputs = nn_outputs.sort_values('targets')

test_df["nn_preds"] = np.array(nn_outputs.preds)
test_df["nn_targets"] = np.array(nn_outputs.targets)
      
### Histogram of ratios
plt.figure(dpi=200)
plt.hist(test_df.nn_preds/test_df.nn_targets, bins=np.linspace(0.5,1.5,40));
plt.xlabel("Test Prediction/Target");
plt.savefig("test_pred_over_target.png");
      
### Response median plot 
import seaborn as sns
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

x = 10**test_df.log10_truth_particle_e
y = test_df.nn_preds/test_df.nn_targets

xbin = [10**exp for exp in np.arange(-1., 3.1, 0.05)]
ybin = np.arange(0., 3.1, 0.05)
xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]
profileXMed = stats.binned_statistic(
    x, y, bins=xbin, statistic='median').statistic

c_map = ListedColormap(sns.color_palette("Blues", n_colors=100).as_hex())
fig = plt.figure(figsize=(12,8), dpi=200)
fig.patch.set_facecolor('white')
plt.hist2d(x, y, bins=[xbin, ybin], norm=LogNorm(),zorder = -1, cmap=c_map);
plt.plot(np.array(xcenter), np.array(profileXMed), color='indianred', linewidth=3)
plt.plot([0.1, 1000], [1, 1], linestyle='--', color='black');
plt.xscale('log')
plt.ylim(0, 1.75)
plt.xlim(0.3, )
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Truth Particle Energy [GeV]', fontsize=20)
plt.ylabel('Predicted Energy / Target', fontsize=20);
np.savez('pub_note_results/response_medians_simple_gnn_test_script_2.npz', response_median=profileXMed, xcenter=xcenter)
plt.savefig("response_median_plot_simple_gnn_test_2.png");
      
### IQR plot 
def iqrOverMed(x):
    # get the IQR via the percentile function
    # 84 is median + 1 sigma, 16 is median - 1 sigma
    q84, q16 = np.percentile(x, [84, 16])
    iqr = q84 - q16
    med = np.median(x)
    return iqr / (2*med)

xbin = [10**exp for exp in np.arange(-1., 3.1, 0.1)]
ybin = np.arange(0., 3.1, 0.05)
xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]

resolution = stats.binned_statistic(x, y, bins=xbin,statistic=iqrOverMed).statistic

fig = plt.figure(figsize=(10,6), dpi=200)
fig.patch.set_facecolor('white')
plt.plot(xcenter, resolution, linewidth=3)
plt.xscale('log')
plt.xlim(0.1, 1000)
plt.ylim(0,0.5)
plt.xlabel('Truth Particle Energy [GeV]', fontsize=20);
plt.ylabel('Response IQR / 2 x Median', fontsize=20);
plt.xticks(fontsize=20);
plt.yticks(fontsize=20);
np.savez('pub_note_results/iqr_simple_gnn_test_script_2.npz', iqr=resolution, xcenter=xcenter)
plt.savefig("iqr_plot_simple_gnn_test_2.png");