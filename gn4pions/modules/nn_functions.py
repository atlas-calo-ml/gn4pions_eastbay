import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

def load_data(file_path = "../data/onetrack_multicluster/pion_files/*.npy", n_files = 100):
    files = glob.glob(file_path)
    df = pd.concat([pd.DataFrame(np.load(file, allow_pickle=True).item()) for file in tqdm(files[:n_files])])
    print("Dataframe has {:,} events.".format(df.shape[0]))
    
    ### Start the dataframe of inputs 
    max_n_clusters = pd.DataFrame(pd.DataFrame(df.cluster_E.to_list())).shape[1]

    df2 = pd.DataFrame(pd.DataFrame(df.cluster_E.to_list(), columns=["cluster_e_"+str(x) for x in np.arange(max_n_clusters)]))
    df2 = pd.DataFrame(df2["cluster_e_0"]) # only keep the leading cluster energy

    df3 = pd.DataFrame(pd.DataFrame(df.cluster_ENG_CALIB_TOT.to_list(), columns=["cluster_calib_e_"+str(x) for x in np.arange(max_n_clusters)]))
    df2["cluster_calib_e_0"] = pd.DataFrame(df3["cluster_calib_e_0"]) # only keep the leading truth cluster energy

    df3 = pd.DataFrame(pd.DataFrame(df.cluster_HAD_WEIGHT.to_list(), columns=["cluster_had_weight_"+str(x) for x in np.arange(max_n_clusters)]))
    df2["cluster_had_weight_0"] = pd.DataFrame(df3["cluster_had_weight_0"]) # only keep the leading truth cluster energy

    ### Add track pT & truth particle E 
    track_pt = np.array(df.trackPt.explode())
    track_eta = np.array(df.trackEta.explode())
    track_phi = np.array(df.trackPhi.explode())
    track_z0 = np.array(df.trackZ0.explode())
    truth_particle_e = np.array(df.truthPartE.explode())
    truth_particle_pt = np.array(df.truthPartPt.explode())

    df2["track_pt"] = track_pt
    df2["track_eta"] = track_eta
    df2["track_phi"] = track_phi
    df2["track_z0"] = track_z0
    df2["truth_particle_e"] = truth_particle_e
    df2["truth_particle_pt"] = truth_particle_pt

    ### Drop infs/NaNs 
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df2 = df2.fillna(0)

    ### Cluster_E > 0.5
    df2 = df2[df2.cluster_calib_e_0 > 0.5]
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

    ### Do this again? 
    df2.replace([np.inf, -np.inf], np.nan, inplace=True)
    df2 = df2.fillna(0)
    
    # ### Inspect variables 
    # sns.set(font_scale = 2)
    # corr_vars = ['truth_particle_e', 'cluster_calib_e_0', 'cluster_e_0', 'track_pt', 'track_eta']
    # g = sns.pairplot(df2[corr_vars], diag_kind='kde')
    # g.fig.set_figheight(12)
    # g.fig.set_figwidth(12)
    
    return(df2)

def regression_model(train_x):
    model = Sequential()
    model.add(Dense(50, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train(df, train_vars, target_var, epochs=10): 
    ### Test/train split 
    train = df.sample(frac=0.8, random_state=0)
    test = df.drop(train.index)

    train_x = train[train_vars].values
    train_y = train[target_var].values
    test_x = test[train_vars].values
    test_y = test[target_var].values

    ### Normalize the inputs 
    sc = StandardScaler()
    train_x = sc.fit_transform(train_x)
    test_x = sc.transform(test_x)
    
    model = regression_model(train_x)
    history = model.fit(
        train_x,
        train_y,
        validation_split=0.2,
        verbose=1, epochs=epochs)
    
    ### Evaluate performance on test set 
    test['nn_target'] = test[target_var]
    test['nn_output'] = model.predict(test_x)
    return(test)

def make_plots(test, plot_em = False, plot_track = False, save_label=None): 
    pred   = test.nn_output
    target = test.nn_target
    x = 10**target
    y = 10**pred/10**target
    
    y_em = test.cluster_e_0/x
#     y_track = test.track_pt/x
    y_track = test.track_pt*np.cosh(test.track_eta)/x
        
    ### Response median plot 
    xbin = [10**exp for exp in np.arange(-1., 3.1, 0.05)]
    ybin = np.arange(0., 3.1, 0.05)
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]
    
    median_em = stats.binned_statistic(x, y_em, bins=xbin, statistic='median').statistic
    median_nn = stats.binned_statistic(x, y, bins=xbin, statistic='median').statistic
    median_track = stats.binned_statistic(x, y_track, bins=xbin, statistic='median').statistic
    
    plt.figure(dpi=100)
    c_map = ListedColormap(sns.color_palette("Blues", n_colors=100).as_hex())
    plt.hist2d(x, y, bins=[xbin, ybin], norm=LogNorm(),zorder = -1, cmap=c_map);
    plt.ylim(0, 1.75)
    plt.plot([0.1, 1000], [1, 1], linestyle='--', color='black')
    plt.plot(xcenter, median_nn, color='indianred')
    plt.xscale('log')

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,5), dpi=100, tight_layout=True)
    fig.patch.set_facecolor('white')

    ax = axs[0]
    ax.plot(np.array(xcenter), np.array(median_nn), color='teal', linewidth=3, label="NN")
    if plot_em: 
        ax.plot(np.array(xcenter), np.array(median_em), color='indianred', linewidth=3, label="EM")
    if plot_track:
        ax.plot(np.array(xcenter), np.array(median_track), color='violet', linewidth=3, label="Track")
    ax.plot([0.1, 1000], [1, 1], linestyle='--', color='black');
    ax.set_xscale('log')
    ax.set_ylim(0.9, 1.2)
    ax.set_xlim(0.3, )
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('True Particle Energy [GeV]', fontsize=15)
    ax.set_ylabel('Predicted Energy / Target', fontsize=15);
    ax.set_title('Response Median', fontsize=20)
    ax.legend(fontsize=15)
    if save_label is not None:
        np.savez('pub_note_results/response_medians_{}.npz'.format(save_label), response_median=median_nn, xcenter=xcenter)
    
    ### IQR plot 
    def iqrOverMed(x):
        # get the IQR via the percentile function
        # 84 is median + 1 sigma, 16 is median - 1 sigma
        q84, q16 = np.percentile(x, [84, 16])
        iqr = q84 - q16
        med = np.median(x)
        return iqr / (2*med)
    
    xbin = [10**exp for exp in np.arange(-1., 3.1, 0.1)]
    ybin = np.arange(0., 3.1, 0.1)
    xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]

    iqr_em = stats.binned_statistic(x, y_em, bins=xbin, statistic=iqrOverMed).statistic
    iqr_nn = stats.binned_statistic(x, y, bins=xbin, statistic=iqrOverMed).statistic
    iqr_track = stats.binned_statistic(x, y_track, bins=xbin, statistic=iqrOverMed).statistic
    
    ax = axs[1]
    ax.plot(xcenter, iqr_nn, linewidth=3, color='teal', label="NN")
    if plot_em: 
        ax.plot(xcenter, iqr_em, linewidth=3, color='indianred', label="EM")
    if plot_track:
        ax.plot(xcenter, iqr_track, linewidth=3, color='violet', label="Track Resolution")
    ax.set_xscale('log')
    ax.set_ylim(0, 0.5)
    ax.set_xlim(0.3, )
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xlabel('True Particle Energy [GeV]', fontsize=15)
    ax.set_ylabel('Response IQR / (2 x Median)', fontsize=15);
    ax.set_title('IQR', fontsize=20)
    ax.legend(fontsize=15)
    if save_label is not None:
        np.savez('pub_note_results/iqr_{}.npz'.format(save_label), iqr=iqr_nn, xcenter=xcenter)