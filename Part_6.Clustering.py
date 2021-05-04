import numpy as np

import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from openpyxl.workbook import Workbook
from mpl_toolkits.mplot3d import axes3d

import sklearn.datasets as dt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
seed = 10
rand_state = 10
filename = input('Введите путь к файлу: ')
# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
plt_ind_list = np.arange(4) + 141
dataset_x=[]
dataset_lab=[]
for std, plt_ind in zip([0.1, 1, 5, 10], plt_ind_list):
    x, label,centers = dt.make_blobs(n_features=2,
                             centers=4,
                             cluster_std=std,
                             random_state=rand_state,
                             return_centers=True)
    dataset_x.append(x)
    dataset_lab.append(label)
    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(x[:, 0],
                                  x[:, 1],
                                  c=label,
                                  vmin=min(label),
                                  vmax=max(label),
                                  cmap=color_map_discrete)
    plt.title('cluster_std: ' + str(std))

nds_x = np.concatenate(dataset_x)
nds_l = np.concatenate(dataset_lab)
ines_x = pd.MultiIndex.from_product([[0.1, 1, 5, 10],np.arange(len(dataset_x[0]))])
cols = pd.MultiIndex.from_product([['centers'],['x','y']])
df_x =pd.DataFrame(nds_x,index=ines_x,columns=['x','y'])
df_l =pd.DataFrame(nds_l,index=ines_x,columns=['label'])
df_c = pd.DataFrame(centers, index=np.arange(len(centers)), columns=cols)
with pd.ExcelWriter(filename) as writer:
    df_x.to_excel(writer)
    df_l.to_excel(writer,startcol=4)
    df_c.to_excel(writer, startcol=8)
fig.subplots_adjust(hspace=0.3, wspace=.3)
plt.suptitle('make_blobs() With Different cluster_std Values', fontsize=20)
plt.show()