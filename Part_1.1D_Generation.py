import numpy as np

import matplotlib.colors
import matplotlib.pyplot as plt
from openpyxl.workbook import Workbook
from mpl_toolkits.mplot3d import axes3d

import sklearn.datasets as dt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

seed = 10
rand_state = 10

color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

rand = np.random.RandomState(seed)

dist_list = ['uniform','normal','exponential','lognormal','chisquare','beta']
param_list = ['-1,1','0,1','1','0,1','2','0.5,0.9']
colors_list = ['green','blue','yellow','cyan','magenta','pink']

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 7))
plt_ind_list = np.arange(6) + 231

filename = input('Введи путь файла: ')
datas = []
for dist, plt_ind, param, colors in zip(dist_list, plt_ind_list, param_list, colors_list):
    x = eval('rand.' + dist + '(' + param + ',5000)')
    a = np.ndarray.tolist(x)
    datas.append(a)
    plt.subplot(plt_ind)
    plt.hist(x, bins=50, color=colors)
    plt.title(dist)
# Вывод в визуал
import pandas as pd
frame = pd.DataFrame(datas, index=dist_list)
with pd.ExcelWriter(filename) as writer:
    frame.to_excel(writer)
fig.subplots_adjust(hspace=0.4, wspace=.3)
plt.suptitle('Sampling from Various Distributions', fontsize=20)
plt.show()