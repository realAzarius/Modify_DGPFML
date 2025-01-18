import copy                                     
import os
import time
import h5py
import torch
from tqdm._tqdm import trange

from clients.clientcfml import ClientCFML
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data


class ServerCFML(Server):

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
            client = ClientCFML(i, train, test, model, num_k, batch_size, inner_lr,
                                outer_lr, local_epochs, test_epochs)  # set cid to int value
            self.clients.append(client)
            self.total_train_examples += client.num_train
            total_test_samples += client.num_test

        print("Finished creating CFML server")

        print("total clients: {}, total train samples: {}, total test samples: {}"
              .format(total_clients, self.total_train_examples, total_test_samples))

    def train(self):
        last_group_res = [0] * len(self.clients)
        current_group_res = [0] * len(self.clients)
        list_dis = []

        for rnd in range(self.num_round):
            optimal_k_set = [[] for i in range(self.K)]
            cluster_discrepancy = .0
            # send $K$ meta model to clients
            self.send_k_model_params()

            if rnd % self.eval_gap == 0 or rnd == self.num_round-1:
                print("---------------- CFML Round ", rnd, "----------------")
                self.evaluate_one_step()
                # for i in current_group_res:
                #     print(i, end='')
                # print()

            # selected clients for training
            self.selected_clients = self.select_clients(
                self.num_select_clients)

            start_time = time.perf_counter()

            for client in self.selected_clients:
                cur_k = client.train()
                current_group_res[client.cid] = cur_k
                optimal_k_set[cur_k].append(client)

            if rnd % self.eval_gap == 0 or rnd == self.num_round-1:
                self.time_per_round.append(time.perf_counter() - start_time)

            # calculate cluster discrepancy
            cluster_discrepancy = sum(
                [0.1 if i != j else 0.0 for i, j in zip(last_group_res, current_group_res)])
            list_dis.append(cluster_discrepancy)
            last_group_res = current_group_res.copy()

            self.aggregate_k_params(optimal_k_set)

            # print('cluster_discrepancy: {}'.format(cluster_discrepancy))

        self.tSNEVisual('tsne_cfml_model_' + self.dataset + '.png')

        self.save_cluster_discrepancy(list_dis)
        self.save_results()
        self.save_k_models()

    def send_k_model_params(self):
        for client in self.clients:
            client.set_k_model_params(self.k_meta_models)

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

    def save_cluster_discrepancy(self, dis):
        alg = self.dataset + '_' + self.algorithm + '_' + str(self.inner_lr) + '_' + \
            str(self.outer_lr) + '_' + str(self.num_select_clients) + \
            'c_' + str(self.num_round) + 'r_' + str(self.K) + 'k'

        with h5py.File("./results/"+'{}_discrepancy.h5'.format(alg), 'w') as hf:
            hf.create_dataset('rs_dis', data=dis)
            hf.close()

    def save_k_models(self):
        model_path = os.path.join("saved_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        for k in range(self.K):
            torch.save(self.model, os.path.join(
                model_path, self.algorithm + "_server_" + str(k) + ".pt"))
