import copy
import time

import torch
from tqdm._tqdm import trange

from clients.clientifca import ClientIFCA
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data
from utils.read_caltech import read_office_caltech
from utils.oh_niid_domain import read_officehome_data

# IFCA
class ServerIFCA(Server):

    def __init__(self, dataset, algorithm, model, k_models, num_k, num_select_clients, batch_size, inner_lr, outer_lr,
                 local_epochs, test_epochs, num_round, eval_gap):
        super().__init__(dataset, algorithm,
                         model[0], num_select_clients, batch_size, inner_lr, outer_lr, local_epochs, num_round)

        # Initialize data for all clients
        total_test_examples = 0
        self.K = num_k

        self.eval_gap = eval_gap
        self.k_models = [copy.deepcopy(model) for model in k_models]

        if dataset in ['office-home', 'office_caltech_10']:
            train_loaders, test_loaders, train_full_loaders, test_full_loaders = read_officehome_data(BATCH_SIZE=batch_size) if dataset == 'office-home' else read_office_caltech(BATCH_SIZE=batch_size)

            total_clients = len(train_loaders)

            for i in trange(total_clients, desc="Create client"):
                client = ClientIFCA(i, [train_loaders[i], train_full_loaders[i]], [test_loaders[i], test_full_loaders[i]], model, self.k_models, batch_size, inner_lr, local_epochs)

                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test
        else:
            data = read_data(dataset)
            total_clients = len(data[0])
            for i in trange(total_clients, desc="Create client"):
                cid, train, test = read_client_data(i, data, dataset)
                client = ClientIFCA(i, train, test, model, self.k_models, batch_size, inner_lr, local_epochs)
                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test

        print("Finished creating IFCA server, total clients: {}, total train samples: {}, total test samples: {}"
              .format(total_clients, self.total_train_examples, total_test_examples))

    def train(self):

        for rnd in trange(self.num_round, desc="Training"):
            optimal_k_set = [[] for i in range(self.K)]

            # send global model to clients
            self.send_k_parameters()
            if rnd % self.eval_gap == 0 or rnd == self.num_round - 1:
                print("---------------- IFCA Round ", rnd, "----------------")
                self.evaluate()
                print()

            # selected clients for training
            self.selected_clients = self.select_clients(
                self.num_select_clients)

            start_time = time.perf_counter()

            for client in self.selected_clients:
                cur_k = client.train()
                optimal_k_set[cur_k].append(client)

            if rnd % self.eval_gap == 0 or rnd == self.num_round - 1:
                self.time_per_round.append(time.perf_counter() - start_time)

            self.aggregate_k_params(optimal_k_set)

        # self.tSNEVisual('tsne_peravg_model_' + self.dataset + '.png')
        self.save_results()
        self.save_k_model()
    
    def save_k_model(self):
        import os
        model_path = os.path.join("saved_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for idx, model in enumerate(self.k_models):
            torch.save(model, os.path.join(model_path, self.algorithm + "_server" + str(idx) + ".pt"))

    def send_k_parameters(self):
        for client in self.clients:
            client.set_k_models(self.k_models)

    def add_k_params(self, k, client, ratio):
        for server_param, client_param in zip(self.k_models[k].parameters(), client.get_model_params()):
            server_param.data += client_param.data.clone() * ratio

    def aggregate_k_params(self, optimal_k_set):
        # get the total number of samples per cluster
        k_total_train = [0] * self.K
        for k, clients in enumerate(optimal_k_set):
            k_total_train[k] = sum([client.num_train for client in clients])

            if len(optimal_k_set[k]) > 0:
                for server_param in self.k_models[k].parameters():
                    server_param.data = torch.zeros_like(server_param.data)

        # aggregate params with ratio
        for k in range(self.K):
            for client in optimal_k_set[k]:
                self.add_k_params(
                    k, client, client.num_train / k_total_train[k])
