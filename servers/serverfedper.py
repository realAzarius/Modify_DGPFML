import time

from tqdm._tqdm import trange

from clients.clientfedper import ClientFedPer
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data
from utils.oh_niid_domain import read_officehome_data


class ServerFedPer(Server):

    def __init__(self, dataset, algorithm, model, num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                 num_round, eval_gap):
        super().__init__(dataset, algorithm, model[0], num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                         num_round)

        self.eval_gap = eval_gap
        total_test_examples = 0

        if dataset == 'office-home':
            train_loaders, test_loaders, train_full_loaders, test_full_loaders = read_officehome_data(BATCH_SIZE=batch_size)

            total_clients = 20
            
            self.eval_gap = eval_gap

            for i in trange(total_clients, desc="Create client"):
                client = ClientFedPer(i, [train_loaders[i], train_full_loaders[i]], [test_loaders[i], test_full_loaders[i]], model, batch_size, inner_lr, local_epochs)

                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test
        else:
            data = read_data(dataset)
            total_clients = len(data[0])
            for i in trange(total_clients, desc="Create client"):
                cid, train, test = read_client_data(i, data, dataset)
                client = ClientFedPer(i, train, test, model, batch_size, inner_lr, local_epochs)
                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test

        print("Finished creating FedPer server, total clients: {}, train samples: {}, test samples: {}".format(
            total_clients, self.total_train_examples, total_test_examples))

    def train(self):

        for rnd in range(self.num_round):
            
            # send global model to clients
            self.send_parameters()
            if rnd % self.eval_gap == 0 or rnd == self.num_round - 1:
                print("---------------- FedPer Round ", rnd, "----------------")
                self.evaluate()

            # selected clients for training
            self.selected_clients = self.select_clients(self.num_select_clients)

            start_time = time.perf_counter()

            for client in self.selected_clients:
                client.train()

            if rnd % self.eval_gap == 0 or rnd == self.num_round - 1:
                self.time_per_round.append(time.perf_counter() - start_time)

            self.aggregate_params()

        self.save_results()
        self.save_model()
