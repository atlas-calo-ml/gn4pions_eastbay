import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from gn4pions.modules.resolution_util import responsePlot,resolutionPlot
from gn4pions.modules.plot_util import roc_plot



def plot_roc_curve(save_path):
    # Define the file paths
    # save_path = "/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/onetrack_multicluster/test_mjg_piplus_pizero_20240308_regress"
    file_path = save_path+'/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)

    # Extract the targets and outputs vectors
    targets = data['targets']
    outputs = data['outputs']
    targets = data['targets']
    outputs = data['outputs']

    print(targets)
    print(len(targets[:,1]))


    auc_score = roc_auc_score(targets[:,1], outputs[:,1])
    print(auc_score)
    fpr, tpr, thresholds = roc_curve(targets[:,1], outputs[:,1])

    # Load the FPR, TPR, and AUC values from files
    # fpr = np.loadtxt(save_path + fpr_file)
    # tpr = np.loadtxt(save_path + tpr_file)
    # with open(save_path +auc_file, 'r') as f:
    #     auc_score = float(f.read())
    
    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.5f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for pi0/piplus')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "_roc_curve.pdf")

def plot_regression(save_path):
    # Define the file paths
    # save_path = "/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/n0_pi0_v04/test_mjg_n0_pi0_20240308_regress"
    # save_path = "/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/onetrack_multicluster/test_mjg_piplus_pizero_20240308_regress"
    file_path = save_path+'/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)

    # Extract the targets and outputs vectors
    targets = data['targets'][:,0]
    outputs = data['outputs'][:,0]
    print(np.mean(outputs))

    


    # print(auc_score)
    # fpr, tpr, thresholds = roc_curve(targets[:,1], outputs[:,1])

    # Load the FPR, TPR, and AUC values from files
    # fpr = np.loadtxt(save_path + fpr_file)
    # tpr = np.loadtxt(save_path + tpr_file)
    # with open(save_path +auc_file, 'r') as f:
    #     auc_score = float(f.read())
    
    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.hist(targets, bins=30, alpha=0.5, color='blue', edgecolor='black',range=(0, 50),label='Truth')
    plt.hist(outputs, bins=30, alpha=0.5, color='red', edgecolor='black',range=(0, 50), label='Predction')

    plt.xlabel('Topocluster E')
    plt.ylabel('Number of events')
    plt.title(f'Energy of TopoCluster')
    plt.legend()
    # plt.grid(True)
    plt.savefig(save_path + "_regression.pdf")
    

def plot_class(save_path):
    # Define the file paths
    # save_path = "/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/onetrack_multicluster/test_mjg_piplus_pizero_20240308_regress"
    file_path = save_path+'/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)

    # Extract the targets and outputs vectors
    targets = data['targets'][:,1]
    outputs = data['outputs'][:,1]
    
    print(np.mean(outputs))


    # print(auc_score)
    # fpr, tpr, thresholds = roc_curve(targets[:,1], outputs[:,1])

    # Load the FPR, TPR, and AUC values from files
    # fpr = np.loadtxt(save_path + fpr_file)
    # tpr = np.loadtxt(save_path + tpr_file)
    # with open(save_path +auc_file, 'r') as f:
    #     auc_score = float(f.read())
    
    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.hist(targets, bins=30, alpha=0.5, color='blue', edgecolor='black',range=(0, 1),label='Truth')
    plt.hist(outputs, bins=30, alpha=0.5, color='red', edgecolor='black',range=(0, 1), label='Predction')

    plt.xlabel('Topocluster E')
    plt.ylabel('Number of events')
    plt.title(f'Energy of TopoCluster')
    plt.legend()
    # plt.grid(True)
    plt.savefig(save_path + "_class.pdf")

# Example: Plot ROC curve for epoch 1

save_path = "/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/n0_pi0_v05/test_mjg_n0_pi0_20240308_regress/"

# plot_roc_curve(save_path)
# plot_regression(save_path)
# plot_class(save_path)

file_path = save_path+'/predictions.npz'

# Load the data from the .npz file
data = np.load(file_path)

# Extract the targets and outputs vectors
targets = data['targets']
outputs = data['outputs']
targets = data['targets']
outputs = data['outputs']

class_truth = data['targets'][:,1]
class_pred  = data['outputs'][:,1]

reg_truth = data['targets'][:,0]
reg_pred = data['outputs'][:,0]
print(targets)
print(len(targets[:,1]))


auc_score = roc_auc_score(class_truth,  class_pred)
print(auc_score)
fpr, tpr, thresholds = roc_curve(class_truth, class_pred)


roc_plot([fpr],[tpr],save_path+"roccy.pdf", atlas_x= .5, atlas_y = .5)

responsePlot(reg_truth , reg_pred/reg_truth, save_path+"reg.pdf")
resolutionPlot(reg_truth , reg_pred/reg_truth, save_path+"res_v01.pdf")
resolutionPlot(reg_truth , reg_pred, save_path+"res_v02.pdf")

# Load the FPR, TPR, and AUC values from files
# fpr = np.loadtxt(save_path + fpr_file)
# tpr = np.loadtxt(save_path + tpr_file)
# with open(save_path +auc_file, 'r') as f:
#     auc_score = float(f.read())

# # Plotting the ROC curve
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.5f}')
# plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title(f'ROC Curve for pi0/piplus')
# plt.legend()
# plt.grid(True)
# plt.savefig(save_path + "_roc_curve.pdf")
