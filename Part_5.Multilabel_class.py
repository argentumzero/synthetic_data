import numpy as np

import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from openpyxl.workbook import Workbook

import sklearn.datasets as dt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
seed = 10
rand_state = 10

filename = input("Введите путь файла: ")
# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(18, 7))
plt_ind_list = np.arange(3) + 131

dataset_x = []
dataset_y = []
dataset_sparse = []
labels = [1,2,4]
for label, plt_ind in zip(labels, plt_ind_list):
    x, y = dt.make_multilabel_classification(n_samples=1000,
                                             n_features=4,
                                             n_labels=label,
                                             n_classes=5,
                                             random_state=rand_state)
    target = np.sum(y * [1,1,1,1,1], axis=1)
    dataset_x.append(x)
    dataset_y.append(y)
    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(x[:, 0],
                                  x[:, 1],
                                  c=target,
                                  vmin=min(target),
                                  vmax=max(target),
                                  cmap=color_map)
    plt.title('n_labels: ' + str(label))
n_ds_x = np.concatenate(dataset_x)
n_ds_y = np.concatenate(dataset_y)
cols_x = pd.MultiIndex.from_product([['x'],[1,2,3,4]])
cols_y = pd.MultiIndex.from_product([['y'],[1,2,3,4,5]])
ines = pd.MultiIndex.from_product([['label_1','label_2','label_4'],np.arange(1000)])
df_x = pd.DataFrame(n_ds_x,columns=cols_x, index=ines)
df_y = pd.DataFrame(n_ds_y,columns=cols_y,index=ines)
with pd.ExcelWriter(filename) as writer:
    df_x.to_excel(writer)
    df_y.to_excel(writer, startcol=7)
fig.subplots_adjust(hspace=0.3, wspace=.3)
plt.suptitle('make_multilabel_classification() With Different n_labels Values', fontsize=20)
plt.show()