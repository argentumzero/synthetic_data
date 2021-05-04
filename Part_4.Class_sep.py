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

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
fig,ax= plt.subplots(nrows=1, ncols=2,figsize=(20, 10))
plt_ind_list = np.arange(4) + 231


filename = input('Введите путь к файлу: ')
dataset_x = []
dataset_y = []
dataset_coef = []
cl_sep = [0.01, 0.1, 1, 10]
schet =0
for class_sep, plt_ind in zip(cl_sep, plt_ind_list):
    x, y = dt.make_classification(n_samples=1000,
                                  n_features=2,
                                  n_repeated=0,
                                  class_sep=class_sep,
                                  n_redundant=0,
                                  random_state=rand_state,)
    dataset_x.append(x)
    dataset_y.append(y)
    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(x[:, 0],
                                  x[:, 1],
                                  c=y,
                                  vmin=min(y),
                                  vmax=max(y),
                                  s=20,
                                  cmap=color_map_discrete)
    plt.title('class_sep: ' + str(class_sep))
ds_x = np.concatenate(dataset_x)
ds_y = np.concatenate(dataset_y)
ines = pd.MultiIndex.from_product([cl_sep,np.arange(1000)])
df_x=pd.DataFrame(ds_x, index= ines,columns=['x1','x2','x3'])
df_y=pd.DataFrame(ds_y, index= ines, columns=['y'])
with pd.ExcelWriter(filename) as writer:
    df_x.to_excel(writer)
    df_y.to_excel(writer, startcol=6)
fig.subplots_adjust(hspace=0.3, wspace=.3)
plt.suptitle('make_classification() With Different class_sep Values', fontsize=20)
plt.show()