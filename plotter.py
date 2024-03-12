import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix

from gn4pions.modules.resolution_util import responsePlot,resolutionPlot
from gn4pions.modules.plot_util import roc_plot

import itertools

import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(save_path):
    # Define the file paths
    file_path = save_path + '/losses.npz'
    
    # Load the loss values from the .npz file
    data = np.load(file_path)
    loss_values_train = data['training']
    loss_values_val = data['validation']
    loss_types = ["","_regress","_class"]

    
    for j in loss_types:
        
        loss_values_train = data['training'+j].mean(axis=1).reshape(-1, 1)
        print(loss_values_train)

        # print(loss_values_train)
        # exit()
        loss_values_val = data['validation'+j].mean(axis=1).reshape(-1, 1)

    # Plotting the loss curve
        plt.figure(figsize=(8, 6))
        epochs = range(1, len(loss_values_train) + 1)  # Assuming loss_values array is 1-indexed for epochs
        plt.plot(epochs, loss_values_train, color='blue', label='Training Loss')
        plt.plot(epochs, loss_values_val, color='red', label='Validation Loss')

        plt.ylim(0, 0.5 * 1.1)  # Setting the y-axis limit a bit higher than the max loss
        plt.xlim(1, 30)  # Setting the x-axis to match the number of epochs
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(j+'Loss Curve over Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path + j+"_loss_curve.pdf")


def plot_roc_curve(save_path):
    # Define the file paths
    file_path = save_path+'/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)

    # Extract the targets and outputs vectors
    targets = data['targets']
    outputs = data['outputs']
    
    auc_score = roc_auc_score(targets[:,1], outputs[:,1])
    fpr, tpr, _ = roc_curve(targets[:,1], outputs[:,1])
    
    # Plotting the ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'AUC = {auc_score:.5f}')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
    plt.ylim(0,1.1)
    plt.xlim(0,1.1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for pi0/piplus')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + "_roc_curve.pdf")



def plot_regression(save_path):
    # Define the file paths
    file_path = save_path+'/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)

    # Extract the targets and outputs vectors
    targets = data['targets'][:,0]
    outputs = data['outputs'][:,0]
    print(np.mean(outputs))

    
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
    file_path = save_path+'/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)

    # Extract the targets and outputs vectors
    targets = data['targets'][:,1]
    outputs = data['outputs'][:,1]
    
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

def plot_cm(save_path,class_0_label,class_1_label):
    # Define the file paths
    file_path = save_path + '/predictions.npz'
    
    # Load the data from the .npz file
    data = np.load(file_path)
    
    # Extract the targets and outputs vectors
    targets = data['targets']
    outputs = data['outputs']
    
    # Assuming outputs are probabilities and we classify as 1 if probability > 0.5
    predicted_labels = np.argmax(outputs, axis=1)
    true_labels = np.argmax(targets, axis=1)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize the confusion matrix by row (i.e., by the number of samples in each true class)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Percentage by True Label)')
    plt.colorbar(format='%0.2f%%')
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, [class_0_label, class_1_label], rotation=45)
    plt.yticks(tick_marks, [class_0_label, class_1_label])
    
    # Labeling the plot
    thresh = cm_percentage.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm_percentage[i, j]:.2f}%",
                 horizontalalignment="center",
                 color="white" if cm_percentage[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot
    plt.savefig(save_path + "_confusion_matrix.pdf")


def plot_response_perClass(save_path,class_0_label,class_1_label):
    # Define the file paths
    
    file_path = save_path + '/predictions.npz'
    
    
    # Load the data from the .npz file
    data = np.load(file_path)
    
    # Extract the targets and outputs vectors
    targets = data['targets'][:,0]
    outputs = data['outputs'][:,0]
    
    truth_labels = data['targets'][:,1]
    
    print(truth_labels)

    class_names = [class_0_label,class_1_label]
    j = 0
    for k in class_names:
        mask = truth_labels == j
        print(mask)
        reg_truth = data['targets'][mask,0]
        reg_pred = data['outputs'][mask,0]
        print(reg_truth)
        print(reg_pred)
        
        responsePlot(reg_truth , reg_pred/reg_truth, save_path+class_names[j]+"_reg.pdf", title="Response Plot for " + k)
        j = j+1
        
    return 

save_path = "/hpcfs/users/a1768536/AGPF/gnn4pions/run_test/results/n0_pi0_03_12_24_v01/test_mjg_n0_pi0_20240312_regress/"
class_0_label = "pi0"
class_1_label = "piplus"
class_1_label = "n0"

plot_loss_curve(save_path)
plot_roc_curve(save_path)
plot_regression(save_path)
plot_class(save_path)
plot_cm(save_path,class_0_label,class_1_label )
plot_response_perClass(save_path,class_0_label,class_1_label)

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
