import sys 
sys.path.append("../../") 
import numpy as np
import matplotlib.pyplot as plt
from utils.model_utils import read_data, read_client_data
from sklearn.manifold import TSNE

def tSNEVisual(save_name, models):
    input_vector = []
    labels = []

    labels = [i for i in range(len(models))]

    for model in models:
        input_vector.append(self.flatten_model_parameters(model.parameters()))


    # Scaling the coordinates to [0, 1]
    def plot_embedding(data):
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        return data

    tsne = TSNE(n_components=2, init='pca', random_state=0, n_jobs=30, verbose=1, n_iter=10000)
    X_tsne = tsne.fit_transform(input_vector)
    aim_data = plot_embedding(X_tsne)

    if self.algorithm == 'PerAvg':
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=4)
        km.fit(aim_data)
        labels = km.labels_.tolist()

    plt.figure()
    plt.subplot(111)
    plt.scatter(aim_data[:, 0], aim_data[:, 1], c=labels)
    plt.savefig(save_name, dpi=600)

# Initialize data for all clients
dataset = 'synthetic'
data = read_data(dataset)
total_clients = len(data[0])
total_test_samples = 0

for i in trange(total_clients, desc="Create client"):
    cid, train, test = read_client_data(i, data, dataset)
    client = ClientOurs(i, train, test, model, batch_size, inner_lr,
                        outer_lr, local_epochs, test_epochs)  # set cid to int value
    self.clients.append(client)
    self.total_train_examples += client.num_train
    total_test_samples += client.num_test
