import matplotlib.colors
import matplotlib.pyplot as plt
import sklearn.datasets as dt
import pandas as pd
import openpyxl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
filename=input("Введите путь: ")
rand_state = 10
pl_ind = np.arange(3)+131
color_map = plt.cm.get_cmap('RdYlBu')
color_map_discrete = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","cyan","magenta","blue"])
labels = [1,2,3]
dataset_x=[]
dataset_y=[]
fig = plt.figure(figsize=(18,5))
for label, plt_ind in zip(labels, pl_ind):
    x, y = dt.make_multilabel_classification(n_samples=20,
                                             n_features=3,
                                             n_labels=label,
                                             n_classes=3,
                                             random_state=rand_state)
    dataset_x.append(x)
    dataset_y.append(y)
    target = np.sum(y*[4,2,1], axis=1)
    ax=fig.add_subplot(plt_ind, projection = '3d',facecolor='black')
    ax.scatter (x[:,0],x[:,1],x[:,2],c=target,s=40, cmap=color_map)
    plt.title('n_labels: ' + str(label))
ines = pd.MultiIndex.from_product([['label1','label2','label3'], np.arange(20)])
df_x = pd.DataFrame(np.concatenate(dataset_x),index=ines, columns=['x','y','z'])
df_y = pd.DataFrame(np.concatenate(dataset_y),index=ines, columns=['class1','class2','class3'])
with pd.ExcelWriter(filename) as writer:
    df_x.to_excel(writer)
    df_y.to_excel(writer, startcol=5)
plt.suptitle('make_multilabel_classification() With Different n_labels Values', fontsize=20)
plt.show()