import copy
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm._tqdm import trange

from clients.clientours import ClientOurs
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data


class ServerOurs(Server):

    def __init__(self, dataset, algorithm, model, k_meta_models, num_k, num_select_clients, batch_size, inner_lr, outer_lr, local_epochs, test_epochs, num_round, eval_gap):
        super().__init__(dataset, algorithm,
                         model[0], num_select_clients, batch_size, inner_lr, outer_lr, local_epochs, num_round)

        self.eval_gap = eval_gap
        self.K = num_k
        self.k_meta_models = [copy.deepcopy(model) for model in k_meta_models]

        # Initialize data for all clients
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

        self.cluster_centers = self.pre_cluster()
        self.client_belong_to_cluster = [0]*len(self.clients)

        print("Finished creating CFML server")

        print("total clients: {}, total train samples: {}, total test samples: {}"
              .format(total_clients, self.total_train_examples, total_test_samples))

    def send_best_parameters(self):
        for client in self.clients:
            client.set_model_params(
                self.k_meta_models[self.client_belong_to_cluster[client.cid]])

    def train(self):
        last_group_res = [0] * len(self.clients)
        current_group_res = [0] * len(self.clients)

        for rnd in range(self.num_round):
            optimal_k_set = [[] for i in range(self.K)]
            cluster_discrepancy = .0
            # send global model to claients
            self.send_best_parameters()
            if rnd % self.eval_gap == 0 or rnd == self.num_round-1:
                print("---------------- Ours Round ", rnd, "----------------")
                self.evaluate_one_step()
                print()

            # selected clients for training
            self.selected_clients = self.select_clients(
                self.num_select_clients)

            start_time = time.perf_counter()

            for client in self.selected_clients:
                cur_k = np.argmax([cosine_similarity(client.train(), j.reshape(1, -1))
                    for j in self.cluster_centers], axis=0).item()
                self.client_belong_to_cluster[client.cid] = cur_k
                current_group_res[client.cid] = cur_k
                optimal_k_set[cur_k].append(client)

            if rnd % self.eval_gap == 0 or rnd == self.num_round-1:
                self.time_per_round.append(time.perf_counter() - start_time)
            
            # calculate cluster discrepancy
            cluster_discrepancy = sum(
                [0.1 if i != j else 0.0 for i, j in zip(last_group_res, current_group_res)])
            last_group_res = current_group_res.copy()

            print('cluster_discrepancy: {}'.format(cluster_discrepancy))

            self.aggregate_k_params(optimal_k_set)
        
        self.save_results()
        self.tSNEVisual('tsne_ours_model_' + self.dataset + '.png')

    def add_k_params(self, k, client, ratio):
        for server_param, client_param in zip(self.k_meta_models[k].parameters(), client.get_model_params()):
            server_param.data += client_param.data.clone() * ratio

    def aggregate_k_params(self, optimal_k_set):
        # get the total number of samples per cluster
        k_total_train = [0]*self.K
        for k, clients in enumerate(optimal_k_set):
            k_total_train[k] = sum([client.num_train for client in clients])

            if len(optimal_k_set[k]) > 0:
                for server_param in self.k_meta_models[k].parameters():
                    server_param.data = torch.zeros_like(server_param.data)

        # aggregate params with ratio
        for k in range(self.K):
            for client in optimal_k_set[k]:
                self.add_k_params(
                    k, client, client.num_train / k_total_train[k])

        # ------------- DEBUF ----------------------------
        # for k, clu in enumerate(optimal_k_set):
        #     print('{}: {}'.format(k, len(clu)))
        # print()
        # ------------------------------------------------

    def pre_cluster(self):
        """Get cluster centers"""
        client_embed = []
        for client in self.clients:
            print('Client {} start.'.format(client.cid))

            client_embed.append(client.get_client_embedding(
                pre_train_epochs=200, dataset=self.dataset))

        km = KMeans(n_clusters=self.K)
        km.fit(client_embed)
        embed_centers = km.cluster_centers_
        labels = km.labels_

        return embed_centers
        # res = [[] for i in range(4)]

        # for i in range(len(client_embed)):
        #     res[labels[i]].append(client_embed[i])

    def test_TSNE(self, input_vector):
        # import numpy as np
        # import matplotlib.pyplot as plt
        # from sklearn.manifold import TSNE

        # # Scaling the coordinates to [0, 1]

        # def plot_embedding(data):
        #     x_min, x_max = np.min(data, 0), np.max(data, 0)
        #     data = (data - x_min) / (x_max - x_min)
        #     return data

        # tsne = TSNE(n_components=2, init='pca', random_state=0,
        #             n_jobs=30, verbose=1, n_iter=10000)
        # X_tsne = tsne.fit_transform(input_vector)
        # aim_data = plot_embedding(X_tsne)
        # plt.figure()
        # plt.subplot(111)
        # plt.scatter(aim_data[:, 0], aim_data[:, 1],
        #             c=[c.cid for c in self.clients])
        # # plt.savefig('./test_TSNE', dpi=600)

        from sklearn import metrics
        import matplotlib.pyplot as plt
        from sklearn.cluster import SpectralClustering
        scores = []
        for i in range(2, 100):
            km = SpectralClustering(n_clusters=i)
            km.fit(input_vector)
            scores.append(metrics.silhouette_score(
                input_vector, km.labels_, metric='euclidean'))
        plt.plot(range(2, 100), scores, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('silhouette_score')
        plt.savefig('./metrics_kmeans.png', dpi=600)
        