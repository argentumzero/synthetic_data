import numpy as np

import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
# from openpyxl.workbook import Workbook
from mpl_toolkits.mplot3d import axes3d

import sklearn.datasets as dt

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
seed = 10
rand_state = 10

# filename = input('Введите путь файла для записи: ')
# Define the color maps for plots
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
fig = plt.figure(figsize=(18,5))

x,y = dt.make_friedman1(n_samples=1000,n_features=5,random_state=rand_state)
dataset_x1=x
dataset_y1=y
ax = fig.add_subplot(131, projection='3d')
my_scatter_plot = ax.scatter(x[:,0], x[:,1],x[:,2], c=y, cmap=color_map)
fig.colorbar(my_scatter_plot)
plt.title('make_friedman1')

x,y = dt.make_friedman2(n_samples=1000,random_state=rand_state)
dataset_x2 = x
dataset_y2 = y
ax = fig.add_subplot(132, projection='3d')
my_scatter_plot = ax.scatter(x[:,0], x[:,1],x[:,2], c=y, cmap=color_map)
fig.colorbar(my_scatter_plot)
plt.title('make_friedman2')

x,y = dt.make_friedman3(n_samples=1000,random_state=rand_state)
dataset_x3 = x
dataset_y3 = y
ax = fig.add_subplot(133, projection='3d')
my_scatter_plot = ax.scatter(x[:,0], x[:,1],x[:,2], c=y, cmap=color_map)
fig.colorbar(my_scatter_plot)
plt.suptitle('make_friedman?() for Non-Linear Data',fontsize=20)
plt.title('make_friedman3')
# df_x1 = pd.DataFrame(dataset_x1, columns=['x0','x1','x2','x3','x4'])
# print(df_x1)
# df_y1 = pd.DataFrame(dataset_y1, columns=['y1'])
# print(df_y1)
# df_x2 = pd.DataFrame(dataset_x2, columns=['x0','x1','x2','x3'])
# print(df_x2)
# df_y2 = pd.DataFrame(dataset_y2,columns=['y2'])
# print(df_y2)
# df_x3 = pd.DataFrame(dataset_x3,columns=['x0','x1','x2','x3'])
# print(df_x3)
# df_y3 = pd.DataFrame(dataset_y3,columns=['y3'])
# print(df_y3)
# with pd.ExcelWriter(filename) as writer:
#     df_x1.to_excel(writer, sheet_name='Sheet1')
#     df_y1.to_excel(writer, sheet_name='Sheet1', startcol=6)
#     df_x2.to_excel(writer, sheet_name='Sheet2')
#     df_y2.to_excel(writer, sheet_name='Sheet2', startcol=5)
#     df_x3.to_excel(writer, sheet_name='Sheet3')
#     df_y3.to_excel(writer, sheet_name='Sheet3', startcol=5)
plt.show()