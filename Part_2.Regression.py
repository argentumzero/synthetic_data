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

filename = input('Введите путь файла: ')

# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 7))
plt_ind_list = np.arange(6) + 231
dataset_x=[]
dataset_y=[]
dataset_coef = []
spise_noise = [0,1,10,100,1000,10000]
for noise, plt_ind in zip(spise_noise, plt_ind_list):
    x, y, coef = dt.make_regression(n_samples=1000,
                              n_features=3,
                              noise=noise,
                              random_state=rand_state,
                                    coef=True)

    plt.subplot(plt_ind)
    dataset_x.append(x)
    dataset_y.append(y)
    dataset_coef.append(coef)
    my_scatter_plot = plt.scatter(x[:, 0],
                                  x[:, 1],
                                  c=y,
                                  vmin=min(y),
                                  vmax=max(y),
                                  s=35,
                                  cmap=color_map)

    plt.title('noise: ' + str(noise))
    plt.colorbar(my_scatter_plot)


dataset_x_new=np.concatenate(dataset_x)
dataset_y_new=np.concatenate(dataset_y)
ines = pd.MultiIndex.from_product([spise_noise,np.arange(1000)])
colms_x = pd.MultiIndex.from_product([['x'],[1,2,3]])
df_x = pd.DataFrame(dataset_x_new,index=ines,columns=colms_x)
df_y = pd.DataFrame(dataset_y_new,index=ines, columns=['y'])
df_coefs = pd.DataFrame(dataset_coef, index=spise_noise, columns=['coef_1','coef_2','coef_3'])
with pd.ExcelWriter(filename) as writer:
    df_x.to_excel(writer)
    df_y.to_excel(writer, startcol=5,startrow=2)
    df_coefs.to_excel(writer, startcol=9, startrow=2)
fig.subplots_adjust(hspace=0.3, wspace=.3)
plt.suptitle('make_regression() With Different Noise Levels', fontsize=20)
plt.show()
