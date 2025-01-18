import time
import os
import torch

from tqdm._tqdm import trange

from clients.clientpfedinit import ClientpFedInit
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data
from utils.oh_niid_domain import read_officehome_data
from utils.read_caltech import read_office_caltech

# pFedInit

class ServerpFedInit(Server):

    def __init__(self, dataset, algorithm, model, num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                 test_epochs, num_round, eval_gap, fixed_weight):
        # model_path = os.path.join("saved_models", dataset, "pFedInit_server.pth")
        # cont = True
        # if os.path.exists(model_path) and cont:
        #     print('Continue training:')
        #     model[0] = torch.load(model_path)
        super().__init__(dataset, algorithm, model[0], num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                         num_round)
        total_test_examples = 0
        self.eval_gap = eval_gap
        self.E = 1

        self.fixed_weight = fixed_weight
        if self.fixed_weight:
            model_name = 'pre_lenet_mnist_fashion_init.pth'
            # model_name = 'pre_alexnet_init.pth'
            # model_name = 'FedAvg_server.pt'
            # model_name = 'pre_lenet_cifar_init.pth'
            # model_name = 'pre_resnet_init.pth'
            path = os.path.join('saved_models', 'pretrain', 'model', model_name)
            # model_name = 'PerAvg_server0.pt'
            # model_name = 'pFedMe_server.pt'
            # path = os.path.join('saved_models', self.dataset, model_name)
            pretrained_model = torch.load(path, map_location=lambda storage, loc: storage)
            pretrained_model_list = []
            for name, param in pretrained_model.items():
                if 'running' in name:
                    continue
                pretrained_model_list.append(param)
            
            global_model_list = []
            for name, param in self.model.named_parameters():
                if 'mtl' in name:
                    continue
                global_model_list.append(param)
            
            for glo, pre in zip(global_model_list, pretrained_model_list):
                glo.data = pre.data.clone()

            print(('*'*10) + '  Global loads sucessfully  ' + ('*'*10))

        if dataset in ['office-home', 'office_caltech_10']:
            train_loaders, test_loaders, train_full_loaders, test_full_loaders = read_officehome_data(BATCH_SIZE=batch_size) if dataset == 'office-home' else read_office_caltech(BATCH_SIZE=batch_size)

            total_clients = len(train_loaders)


            for i in trange(total_clients, desc="Create client"):
                client = ClientpFedInit(i, [train_loaders[i], train_full_loaders[i]], [test_loaders[i], test_full_loaders[i]], model, batch_size, inner_lr, outer_lr, local_epochs, test_epochs, fixed_weight, self.E)

                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test
        else:                
            # Initialize data for all clients
            data = read_data(dataset)
            total_clients = len(data[0])

            for i in trange(total_clients, desc="Create client"):
                cid, train, test = read_client_data(i, data, dataset)
                client = ClientpFedInit(i, train, test, model, batch_size, inner_lr, outer_lr, local_epochs, test_epochs, fixed_weight, self.E)

                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test

        print("Finished creating pFedInit server, total clients: {}, total train samples: {}, total test samples: {}"
              .format(total_clients, self.total_train_examples, total_test_examples))

    def train(self):

        for rnd in trange(self.num_round, desc="Training"):
            update = True if (rnd+3) % self.E == 0 else False
            # send global model to clients
            if self.fixed_weight:
                self.send_classifier_parameters()
            else:
                self.send_feature_parameters(features=update)

            if rnd % self.eval_gap == 0 or rnd == self.num_round - 1:
                print("---------------- pFedInit Round ", rnd, "----------------")
                self.evaluate_one_step()
                print()

            # selected clients for training
            self.selected_clients = self.select_clients(self.num_select_clients)

            start_time = time.perf_counter()

            for client in self.selected_clients:
                client.train(global_comm=rnd)

            if rnd % self.eval_gap == 0 or rnd == self.num_round - 1:
                self.time_per_round.append(time.perf_counter() - start_time)

            self.aggregate_params_pfedinit(features=update)

        self.save_results()
        self.save_model()
        self.save_personalized_model()

    def aggregate_params_pfedinit(self, features):
        """Aggregate selected clients' model parameters"""
        # set server model parameters to zero
        for name, server_param in self.model.named_parameters():
            if ('classifier' in name and self.fixed_weight) or (features and 'features' in name and 'mtl' not in name):
                server_param.data = torch.zeros_like(server_param.data)

        total_train = 0
        for client in self.selected_clients:
            total_train += client.num_train
        for client in self.selected_clients:
            for server_params, client_param in zip(self.model.named_parameters(), client.get_model_params()):
                if ('classifier' in server_params[0] and self.fixed_weight) or (features and 'features' in server_params[0] and 'mtl' not in server_params[0]):
                    server_params[1].data += client_param.data.clone() * (client.num_train / total_train)

    def send_classifier_parameters(self):
        for client in self.clients:
            client_model = client.model
            client_local_model = client.local_model
            for old_params, new_param, local_param in zip(client_model.named_parameters(), self.model.parameters(), client_local_model.parameters()):
                if 'classifier' in old_params[0]:
                    old_params[1].data = new_param.data.clone()
                    local_param.data = new_param.data.clone()

    def send_feature_parameters(self, features=True):
        for client in self.clients:
            client_model = client.model
            client_local_model = client.local_model
            for old_params, new_param, local_param in zip(client_model.named_parameters(), self.model.parameters(), client_local_model.parameters()):
                if ('classifier' in old_params[0] and self.fixed_weight) or (features and 'features' in old_params[0] and 'mtl' not in old_params[0]):
                    old_params[1].data = new_param.data.clone()
                    local_param.data = new_param.data.clone()

    def save_personalized_model(self):
        model_path = os.path.join("saved_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        for client in self.clients:
            client.train_one_step()
            torch.save(client.model, os.path.join(model_path, self.algorithm + "_server" + str(client.cid) + ".pt"))
