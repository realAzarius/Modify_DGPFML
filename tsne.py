import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# 将坐标缩放到[0,1]区间  
def plot_embedding(data):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    return data


input_vector = []
labels = []

for cid in range(100):
    model_path = os.path.join('./saved_models', 'synthetic4', 'pFedInit_server' + str(cid) + '.pt')
    if os.path.exists(model_path):
        model = torch.load(model_path)
        modellist = []
        for param in model.parameters():
            modellist.append(param)
        
        vec = []
        flag = 0
        i = 0
        n = len(modellist)
        while i < n-4:
            new_weight = modellist[i].mul(modellist[i+1].expand(modellist[i].shape)).data.flatten().numpy()
            
            new_bias = modellist[i+2].add(modellist[i+3]).repeat(modellist[i].shape[1]).data.flatten().numpy()
            vec = vec + (new_weight + new_bias).tolist()
                
            i += 4
        # vec = []
        # for param in model.parameters():
        #     vec = vec + param.data.flatten().numpy().tolist()
        input_vector.append(vec)
# 可视化参数

# tsne = TSNE(n_components=3, init='pca', random_state=0, n_jobs=30, verbose=1, n_iter=20000)
# X_tsne = tsne.fit_transform(input_vector)
# aim_data = plot_embedding(X_tsne)
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(211, projection='3d')
# ax.scatter(aim_data[:, 0], aim_data[:, 1], aim_data[:, 2], c=labels, cmap=plt.cm.Spectral)
# ax.view_init(4, -72)
# plt.title("T-SNE Digits")
# plt.savefig('./tSNE8_1.png')

tsne = TSNE(n_components=2, init='pca', random_state=0, n_jobs=30, verbose=1, n_iter=20000)
X_tsne = tsne.fit_transform(input_vector)
aim_data = plot_embedding(X_tsne)

# K-means
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=10007).fit_predict(aim_data)



fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
ax.scatter(aim_data[:, 0], aim_data[:, 1],c = y_pred, cmap=plt.cm.Spectral)
plt.title("T-SNE")
plt.savefig('./tSNE3.png', dpi=600)