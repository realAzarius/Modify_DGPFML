import copy
import random
import numpy as np
import torch
import tqdm
import os
import h5py
from sklearn.cluster import DBSCAN, KMeans
#from sklearn.cluster import kmeans_plusplus
#from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.std import trange
from clients.clientours import ClientOurs
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data
from models.task_embedding import Autoencoder
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler




class ServerOurs(Server):

    def __init__(self, dataset, algorithm, model, k_meta_models, num_k, num_select_clients, batch_size, inner_lr, outer_lr, local_epochs, test_epochs, num_round, eval_gap):
        super().__init__(dataset, algorithm,
                         model[0], num_select_clients, batch_size, inner_lr, outer_lr, local_epochs, num_round)
        self.client_belong_to_cluster = []
        self.dataset = dataset
        self.dynamic_cluster = True
        self.global_average_ae = False
        self.model_name = model[1]
        self.eval_gap = eval_gap
        #self.best_k =None
        #self.num_k = self.best_k
        self.k_meta_models = [copy.deepcopy(model) for model in k_meta_models]
        self.max_k = 50
        self.optimal_k_list = []
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
        self.cluster_centers = self.pre_cluster(self.dataset)
        #self.cluster_centers = [list(center) for center in self.cluster_centers]
        if isinstance(self.cluster_centers, int):

            self.cluster_centers = [self.cluster_centers]
        elif not isinstance(self.cluster_centers, list):

            print("Warning: self.cluster_centers should be a list. Setting it to an empty list.")
            self.cluster_centers = []


        self.cluster_centers = [list(center) for center in self.cluster_centers]
        self.client_belong_to_cluster = [0]*len(self.clients)

        print("Finished creating Ours server")

        print("total clients: {}, total train samples: {}, total test samples: {}"
              .format(total_clients, self.total_train_examples, total_test_samples))


    def send_best_parameters(self):
        for client in self.clients:
            client.set_model_params(self.k_meta_models[self.client_belong_to_cluster[client.cid]])

    def select_clients(self, num_clients):
        """selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))

        Return:
            list of selected clients objects
        """
        if num_clients == len(self.clients):
            print("All users are selected")
            return self.clients

        num_clients = min(num_clients, len(self.clients))
        return np.random.choice(self.clients, num_clients, replace=False)

    def test(self):
        """Get test results from clients

        Returns:
            cids: list of client's id
            num_samples: list of client's test size
            tot_correct: the number of accurate predictions per client
        """
        num_samples = []
        tot_correct = []

        for c in self.clients:
            acc, ns = c.test()
            tot_correct.append(acc * 1.0)
            num_samples.append(ns)

        cids = [c.cid for c in self.clients]

        return cids, num_samples, tot_correct

    def train_error_and_loss(self):
        num_samples = []
        tot_correct = []
        losses = []

        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss()
            tot_correct.append(ct * 1.0)
            num_samples.append(ns)
            losses.append(cl * 1.0)

        cids = [c.cid for c in self.clients]

        return cids, num_samples, tot_correct, losses

    def evaluate(self):
        stats_test = self.test()

        if self.dataset != "office-home":
            stats_train = self.train_error_and_loss()

        if self.dataset == "office_caltech_10":
            test_loaders = []
            for c in self.clients:
                test_loaders.append(c.test_loader_full)

            for cid, c in enumerate(self.clients):
                cmodel = c.model
                for tid, test_loader in enumerate(test_loaders):
                    cmodel.eval()
                    cmodel.to("cpu")
                    test_acc = 0
                    num_test = 0
                    with torch.no_grad():
                        for X, y in test_loader:
                            num_test = len(y)
                            output = cmodel(X)
                            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_acc = test_acc / num_test
                    content = str(cid) + "in " + str(tid) + " acc: " + str(test_acc) + "\n"
                    file_path = './' + self.algorithm + '_domain.txt'
                    with open(file_path, 'a+') as fp:
                        fp.write(content)
                file_path = './' + self.algorithm + '_domain.txt'
                with open(file_path, 'a+') as fp:
                    fp.write("\n")

        # The average testing accuracy rate of all clients
        global_acc = np.sum(stats_test[2]) * 1.0 / np.sum(stats_test[1])

        # The average training accuracy rate of all clients
        client_acc = np.array(stats_test[2]) / np.array(stats_test[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1]) if self.dataset != "office-home" else []
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(
            stats_train[1]) if self.dataset != "office-home" else []

        self.client_acc.append(client_acc)
        self.rs_glob_acc.append(global_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)
        print("Average Global Testing Accurancy: ", global_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)

    def evaluate_one_step(self):
        cluster_centers, random_k = self.pre_cluster(self.dataset)
        self.best_k = random_k
        if self.dataset == "office_caltech_10":
            test_loaders = []
            for c in self.clients:
                test_loaders.append(c.test_loader_full)
                c.train_one_step()
            stats_test = self.test()
            for cid, c in enumerate(self.clients):
                cmodel = c.model
                for tid, test_loader in enumerate(test_loaders):
                    if self.algorithm == "Ours":
                        c.train_one_step(test_loader)

                    cmodel.eval()
                    cmodel.to("cpu")
                    test_acc = 0
                    num_test = 0
                    with torch.no_grad():
                        for X, y in test_loader:
                            num_test = len(y)
                            output = cmodel(X)
                            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    test_acc = test_acc / num_test

                    content = str(cid) + " in " + str(tid) + " acc: " + str(test_acc) + "\n"
                    file_path = './' + self.algorithm + '_domain.txt'
                    with open(file_path, 'a+') as fp:
                        fp.write(content)
                    if self.algorithm == "Ours":
                        c.update_parameters(c.local_model.parameters())

                file_path = './' + self.algorithm + '_domain.txt'
                with open(file_path, 'a+') as fp:
                    fp.write("\n")

        for c in self.clients:
            c.train_one_step()
        stats_test = self.test()

        if self.dataset not in ["office-home", 'office_caltech_10']:
            stats_train = self.train_error_and_loss()

        # recover meta-model parameters with local model parameters
        for c in self.clients:
            c.update_parameters(c.local_model.parameters())

        client_acc = np.array(stats_test[2]) / np.array(stats_test[1])
        glob_acc = np.sum(stats_test[2]) * 1.0 / np.sum(stats_test[1])
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1]) if self.dataset not in ["office-home",
                                                                                                  'office_caltech_10'] else []
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(
            stats_train[1]) if self.dataset not in ["office-home", 'office_caltech_10'] else []
        self.client_acc.append(client_acc)
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)
        self.optimal_k_list.append(random_k)
        if len(self.rs_glob_acc_per) == 1200:
           avg_test = sum(self.rs_glob_acc_per) / len(self.rs_glob_acc_per)
           avg_train = sum(self.rs_train_acc_per) / len(self.rs_train_acc_per)
           print("Avg test: ", avg_test)
           print("Avg train: ", avg_train)
        print("Average Personal Testing Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)
        print("Optimal K:", random_k)
    def send_parameters(self):
        """
        Send server model to all clients
        """
        for client in self.clients:
            client.set_model_params(self.model)

    # Save loss, accurancy to h5 file
    def save_results(self):
        alg = self.dataset + '_' + self.algorithm + '_' + str(self.batch_size) + '_' + str(
            self.local_epochs) + '_' + str(self.inner_lr) + '_' + \
              str(self.outer_lr) + '_' + str(self.num_select_clients) + 'c_' + str(self.num_round) + 'r'

        if self.algorithm == 'CFML' or self.algorithm == 'Ours':
            alg += '_'
        #if self.algorithm == 'CFML' or self.algorithm == 'Ours':
            #alg += '_' + str(self.random_k) + 'k'

        if (len(self.rs_glob_acc) != 0 and len(self.rs_train_acc) and len(self.rs_train_loss)):
            with h5py.File("./results/" + '{}.h5'.format(alg), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_round_time', data=self.time_per_round)
                hf.create_dataset('rs_client_acc', data=self.client_acc)
                hf.close()

        if (len(self.rs_glob_acc_per) != 0 and len(self.rs_train_acc_per) and len(self.rs_train_loss_per)):
            with h5py.File("./results/" + '{}.h5'.format(alg), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.create_dataset('rs_round_time', data=self.time_per_round)
                hf.create_dataset('random_k2', data=self.optimal_k_list)
                hf.create_dataset('rs_client_acc', data=self.client_acc)
                hf.close()

    def save_model(self):
        model_path = os.path.join("saved_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, self.algorithm + "_server" + ".pt"))

    def flatten_model_parameters(self, parameters):
        return torch.cat([x.flatten() for x in parameters], 0).detach().numpy().tolist()

    def train(self):
        self.optimal_k_list = []

        for rnd in range(self.num_round):

            cluster_centers, random_k = self.pre_cluster(self.dataset)
            self.optimal_k_list.append(random_k)

            self.best_k = random_k
            self.num_k = random_k
            self.cluster_centers = cluster_centers
            optimal_k_set = [[] for _ in range(random_k)]
            self.send_best_parameters()
            if rnd % self.eval_gap == 0 or rnd == self.num_round-1:
                print("---------------- Ours Round ", rnd, "----------------")
                self.evaluate_one_step()
                print()

            # selected clients for training
            self.selected_clients = self.select_clients(
                self.num_select_clients)
            for client in self.selected_clients:

                #cur_clusters = [self.cluster_centers[i] for i in self.client_belong_to_cluster]
                if self.dynamic_cluster:

                    cluster_idx = self.client_belong_to_cluster[client.cid]
                    if isinstance(cluster_idx, int) or isinstance(cluster_idx, np.int64):
                        cluster_idx = [cluster_idx]
                    else:
                        cluster_idx = cluster_idx.tolist()


                    cur_clusters = [self.cluster_centers[i] for i in cluster_idx]


                    cur_k = np.argmax(
                        [cosine_similarity(client.train(), np.tile(j, (client.train().shape[0], 1))).flatten()
                         for j in cur_clusters])
                    self.client_belong_to_cluster[client.cid] = cur_k


                    if cur_k >= len(optimal_k_set):
                        optimal_k_set.extend([[] for _ in range(cur_k - len(optimal_k_set) + 1)])
                    optimal_k_set[cur_k].append(client)
                else:
                    # Use fixed clustering
                    cur_k = self.client_belong_to_cluster[client.cid]
                    optimal_k_set[cur_k].append(client)
                self.aggregate_k_params(optimal_k_set)

                self.save_results()


    def add_k_params(self, k, client, ratio):
        for server_param, client_param in zip(self.k_meta_models[k].parameters(), client.get_model_params()):
            server_param.data += client_param.data.clone() * ratio

    def aggregate_k_params(self, optimal_k_set):
        k_total_train = [0] * (self.max_k + 1)
        #k_total_train = [0] * (self.num_k + 1)
        for k, clients in enumerate(optimal_k_set):
            k_total_train[k] = sum([client.num_train for client in clients])
            if len(optimal_k_set[k]) > 0:
                for server_param in self.k_meta_models[k].parameters():
                    server_param.data = torch.zeros_like(server_param.data)
        # aggregate params with ratio

        for k in range(self.num_k):
            if len(optimal_k_set) <= k:
                optimal_k_set.append([])
            #assert k >= 0 and k < len(optimal_k_set), f"Invalid index: {k}"
            for client in optimal_k_set[k]:
                self.add_k_params(k, client, client.num_train / k_total_train[k])

    def pre_cluster(self,dataset):
        """
        Get cluster centers
        Return: list of embedding centers
        """
        num_iterations= 1200
        client_embed = []

        if self.model_name == "lstm":
            global_ae = Autoencoder(input_size=80, embedding_size=8)
        elif self.model_name == "mclr":
            global_ae = Autoencoder(input_size=60, embedding_size=6)
        else:
            global_ae = Autoencoder(input_size=784, embedding_size=64)

        if self.global_average_ae:
            # average global AE
            for server_param in global_ae.parameters():
                server_param.data = torch.zeros_like(server_param.data)
            total_train = sum([client.num_train for client in self.clients])
            for client in tqdm.tqdm(self.clients):
                for server_param, client_param in zip(global_ae.parameters(), client.embed_model.parameters()):
                    server_param.data += client_param.data.clone() * (client.num_train / total_train)
            # send ae to all clients
            for client in self.clients:
                for old_param, new_param in zip(client.embed_model.parameters(), global_ae.parameters()):
                    old_param.data = new_param.data.clone()

        for client in self.clients:
            client_embed.append(client.get_client_embedding())
        client_embed = np.asarray(client_embed)

        scaler = StandardScaler()
        client_embed = scaler.fit_transform(client_embed)

        eps_values = [1,2,3]
        min_samples_values = [5,7,8]
        #eps_values = [i for i in [0.01, 0.1, 1, 2, 3,5,6,7,8,9, 10] if 0.01 <= i <= 10]
        #min_samples_values = [i for i in [2, 3, 4, 5, 6, 7, 8,9,10,11,12,13,14,15,16,17,18,19,20] if 2 <= i <= 20]
        best_k = -1
        best_silhouette_score = -1
        best_labels = []

        for _ in range(num_iterations):
            eps = random.choice(eps_values)
            min_samples = random.choice(min_samples_values)
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan_labels = dbscan.fit_predict(client_embed)
            unique_labels = set(dbscan_labels)

            if len(unique_labels) <= 1:
                continue


            silhouette_avg = silhouette_score(client_embed, dbscan_labels)

            if silhouette_avg > best_silhouette_score:
                    best_silhouette_score = silhouette_avg
                    best_k = len(unique_labels)
                    best_labels = dbscan_labels
                    best_eps = eps
                    best_min_samples = min_samples

        if best_k == -1:
            raise ValueError("Unable to determine optimal K value for DBSCAN")



        if len(best_labels) > 0:
           core_samples_mask = np.zeros_like(best_labels, dtype=bool)
           core_samples_mask[best_labels == 1] = True
           core_samples_mask = np.asarray(core_samples_mask, dtype=int)
           dbscan_clusters = client_embed[core_samples_mask]

        random_k = random.choice(range(2, len(unique_labels) + 1))

        kmeans = KMeans(n_clusters=random_k)

        kmeans.fit(dbscan_clusters)
        embed_centers = kmeans.cluster_centers_

        # Set the self.num_k value
        self.num_k = random_k
        self.optimal_k_list.append(random_k)
        #print(self.num_k)
        #print(best_k)
        return embed_centers, random_k


'''
            # Run K-Means
            km = KMeans(n_clusters=self.K)
            km.fit(client_embed)
            embed_centers = km.cluster_centers_
            # init self.client_belong_to_cluster
            self.client_belong_to_cluster = km.labels_.tolist()

            return embed_centers
'''