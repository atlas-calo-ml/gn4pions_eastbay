import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import seaborn as sns
import pickle
sns.set_context('poster')
import glob
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay #, plot_roc_curve
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wasserstein_distance
import os
import gn4pions.modules.resolution_util as ru
import gn4pions.modules.plot_util as pu
import atlas_mpl_style as ampl
ampl.use_atlas_style(usetex=True)
plt.style.use('print')
ampl.set_color_cycle('Oceanic',10)


median_gnn = np.load('/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/n0_pi0_alpha_0.5_v02/test_mjg_n0_pi0_20240308_regress/predictions.npz')
# median_em =  np.load('pub_note_results/response_medians_clusteronly_em.npz')
# median_lcw =  np.load('pub_note_results/response_medians_clusteronly_lcw.npz')

from scipy.signal import savgol_filter # for smoothing the PFN results due to stats difference

# xcenter = median_gnn['xcenter']
median_2 = [
    median_gnn['response_median'][0,:],
    # median_lcw['response_median'][0,:],
    # median_gnn['response_median'][0,:],
      ]
labels=['EM', 'LCW', 'GNN',]

### Response medians 
pu.lineOverlay(xcenter=xcenter, lines=median_2,  
            labels=labels,
            xlabel = 'True Cluster Energy [GeV]', ylabel = 'Response Median',
            figfile = 'pub_note_results/regress_response_medians_all_clusteronly.pdf',
            y_max=1.2, y_min=.9, 
               x_min = .3, 
               colorgrouping=0,
            extra_lines= [[[0, 10e3], [1, 1]]],
            linestyles = ['dashed', 'dashed', 'solid', 'solid', 'solid', 'solid'],
            atlas_x = 0.45, atlas_y = 0.7, simulation = True,
            textlist = [{'x': 0.45, 'y': 0.6, 'text': 'Single $\pi^0/\pi^{\pm}$ MC Regression'},
                        {'x': 0.45, 'y': 0.55,  'text': 'Topo-clusters, |$\eta$| < 3'},])