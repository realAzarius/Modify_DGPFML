import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


rootdir = './'
NUM_COLORS = 20
lst = os.listdir(rootdir)
embeddings = []
client_ids = []
colors = ['black', 'silver', 'lightcoral', 'red', 'tomato',
          'sienna', 'peachpuff', 'linen', 'darkorange', 'tan',
          'gold', 'darkkhaki', 'dodgerblue', 'yellow', 'darkolivegreen',
          'navy', 'darkcyan', 'violet', 'fuchsia', 'crimson']

super_class = ['aquatic mammals',
'fish',
'flowers',
'food',
'fruit and vegetables',
'household electrical devices',
'household',
'insects',
'large carnivores',
'large man-made outdoor things',
'large natural outdoor scenes',
'large omnivores and herbivores',
'medium-sized mammals',
'non-insect invertebrates',
'people',
'reptiles',
'small mammals',
'trees',
'vehicles 1',
'vehicles 2']
epoch = 'E5000_result'
labels=[]

cm = plt.get_cmap('gist_rainbow')

def tSNEVisual(save_name, input_vector, labels, client_ids):

        def plot_embedding(data):
            x_min, x_max = np.min(data, 0), np.max(data, 0)
            data = (data - x_min) / (x_max - x_min)
            return data
        
        tsne = TSNE(n_components=2, init='pca', random_state=random.randint(1, 1000000), n_jobs=30, verbose=1, n_iter=200000)
        X_tsne = tsne.fit_transform(input_vector)
        aim_data = plot_embedding(X_tsne)

        plt.figure(figsize=(12, 5))
        plt.subplot(111)
        s = 0
        for (idx, col) in enumerate(colors):
            idxs = [i for i in range(len(labels)) if labels[i] == matplotlib.colors.cnames[col]]
            plt.scatter(aim_data[idxs, 0], aim_data[idxs, 1], c=matplotlib.colors.cnames[col], edgecolors='k', label=super_class[idx])
            s += len(idxs)
        print(s)
        # plt.scatter(aim_data[:, 0], aim_data[:, 1], c=labels, edgecolors='k')
        plt.subplots_adjust(right=0.7)
        plt.legend(loc=2, bbox_to_anchor=(1.02,1.0),borderaxespad = 0.)
        plt.savefig(save_name, dpi=600)

for line in lst:
    filepath = os.path.join(rootdir, line)
    if os.path.isfile(filepath) and filepath.endswith(".txt") and epoch in filepath:
        with open(filepath, 'r') as fp:
            line = fp.readline().split(',')
            emb = list(map(float, line))
            embeddings.append(torch.FloatTensor(emb).numpy().tolist())

            cid = int(filepath.replace('./C', '').split('_')[0])-1
            # labels.append(cm(1.*(cid//5)/NUM_COLORS+0.2))
            labels.append(matplotlib.colors.cnames[colors[cid//5]])
            client_ids.append(cid)

print('Total clients: ', len(embeddings))
tSNEVisual('./test_' + epoch + '.pdf', embeddings, labels, client_ids)