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
filename = input('Введите путь к файлу: ')
# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))
plt_ind_list = np.arange(4) + 141
dataset_x=[]
dataset_lab=[]
for noise, plt_ind in zip([0, 0.1, 0.5, 1], plt_ind_list):
    x, label = dt.make_circles(noise=noise, random_state=rand_state)
    dataset_x.append(x)
    dataset_lab.append(label)
    plt.subplot(plt_ind)
    my_scatter_plot = plt.scatter(x[:, 0],
                                  x[:, 1],
                                  c=label,
                                  vmin=min(label),
                                  vmax=max(label),
                                  cmap=color_map_discrete)
    plt.title('noise: ' + str(noise))
nds_x = np.concatenate(dataset_x)
nds_l = np.concatenate(dataset_lab)
ines_x = pd.MultiIndex.from_product([[0, 0.1, 1, 2],np.arange(len(dataset_x[0]))])
cols_x = pd.MultiIndex.from_product([['x'],[1,2]])
df_x =pd.DataFrame(nds_x,index=ines_x,columns=cols_x)
df_l =pd.DataFrame(nds_l,index=ines_x,columns=['label'])
with pd.ExcelWriter(filename) as writer:
    df_x.to_excel(writer)
    df_l.to_excel(writer,startcol=4,startrow=2)
fig.subplots_adjust(hspace=0.3, wspace=.3)
plt.suptitle('make_circles() With Different Noise Levels', fontsize=20)
plt.show()