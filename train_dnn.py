import pandas as pd
import numpy as np
import glob
import os
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from keras import callbacks
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2" 
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### Load data
train = pd.read_hdf("train_dnn.h5")
val = pd.read_hdf("val_dnn.h5")
test = pd.read_hdf("test_dnn.h5")

### All clusters + track eta/phi/z0
train_vars = [var for var in test.keys() if var.startswith('log10') and var != 'log10_truth_particle_e']
train_vars += ['track_eta', 'track_phi', 'track_z0']

print("Training variables: {}".format(train_vars))

train_x = train[train_vars].values
train_y = train['log10_truth_particle_e'].values
val_x = val[train_vars].values
val_y = val['log10_truth_particle_e'].values
test_x = test[train_vars].values
test_y = test['log10_truth_particle_e'].values

### Normalize the inputs 
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
val_x = sc.transform(val_x)
test_x = sc.transform(test_x)

def regression_model():
    model = Sequential()
    model.add(Dense(64, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model

model = regression_model()
print(model.summary())

early_stopping = callbacks.EarlyStopping(monitor='val_loss', 
                                         patience=30, 
                                         verbose=0) 

### Save the best weights
weights_path = os.path.join("dnn_best_weights_64.h5")
checkpoint = callbacks.ModelCheckpoint(weights_path, 
                                       monitor='loss', 
                                       mode='auto', 
                                       verbose=0, 
                                       save_best_only=True, 
                                       save_weights_only=True)
history = model.fit(
    train_x,
    train_y,
    validation_data = (val_x, val_y),
    verbose=1, epochs=300, 
    batch_size=1024,
    callbacks=[early_stopping, checkpoint])

### Evaluate performance on test set 
test['nn_output'] = model.predict(test_x)

### Response median plot 
import scipy.stats as stats
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

x = test.truth_particle_e
y = 10**test.nn_output/test.truth_particle_e

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
plt.xlabel('Truth Particle Energy [GeV]')
plt.ylabel('Predicted Energy / Target');
np.savez('pub_note_results/response_median_dnn_64.npz', response_median=profileXMed, xcenter=xcenter)
plt.savefig('pub_note_results/response_median_dnn_64.png')

### IQR plot 

def iqrOverMed(x):
    # get the IQR via the percentile function
    # 84 is median + 1 sigma, 16 is median - 1 sigma
    q84, q16 = np.percentile(x, [84, 16])
    iqr = q84 - q16
    med = np.median(x)
    return iqr / (2*med)

xbin = [10**exp for exp in np.arange(-1., 3.1, 0.1)]
xcenter = [(xbin[i] + xbin[i+1]) / 2 for i in range(len(xbin)-1)]

resolution = stats.binned_statistic(x, y, bins=xbin,statistic=iqrOverMed).statistic

fig = plt.figure(figsize=(10,6), dpi=200)
fig.patch.set_facecolor('white')
plt.plot(xcenter, resolution, linewidth=3)
plt.xscale('log')
plt.xlim(0.1, 1000)
plt.ylim(0,0.5)
plt.xlabel('Truth Particle Energy [GeV]')
plt.ylabel('Response IQR / 2 x Median');

np.savez('pub_note_results/iqr_dnn_64.npz', iqr=resolution, xcenter=xcenter)
plt.savefig('pub_note_results/iqr_dnn_64.png')
