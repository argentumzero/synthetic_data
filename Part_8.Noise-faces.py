import numpy as np

import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from openpyxl.workbook import Workbook
from mpl_toolkits.mplot3d import axes3d

import sklearn.datasets as dt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
seed = 11
rand_state = 11

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
# Fetch the dataset and store in X
faces = dt.fetch_olivetti_faces()
X= faces.data

# Fit a kernel density model using GridSearchCV to determine the best parameter for bandwidth
bandwidth_params = {'bandwidth': np.arange(0.01,1,0.05)}
grid_search = GridSearchCV(KernelDensity(), bandwidth_params)
grid_search.fit(X)
kde = grid_search.best_estimator_

# Generate/sample 8 new faces from this dataset
new_faces = kde.sample(8, random_state=rand_state)

# Show a sample of 8 original face images and 8 generated faces derived from the faces dataset
fig,ax = plt.subplots(nrows=2, ncols=8,figsize=(18,6),subplot_kw=dict(xticks=[], yticks=[]))
for i in np.arange(8):
    ax[0,i].imshow(X[10*i,:].reshape(64,64),cmap=plt.cm.gray)
    ax[1,i].imshow(new_faces[i,:].reshape(64,64),cmap=plt.cm.gray)
ax[0,3].set_title('Original Data',fontsize=20)
ax[1,3].set_title('Synthetic Data',fontsize=20)
fig.subplots_adjust(wspace=.1)
plt.show()