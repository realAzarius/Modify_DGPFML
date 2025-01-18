import torch
import copy
import numpy as np
from clients.clientpfedme import clientpFedMe
from servers.serverbase import Server
from utils.model_utils import read_data, read_client_data
from utils.oh_niid_domain import read_officehome_data
from utils.read_caltech import read_office_caltech
from tqdm._tqdm import trange
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class serverpFedMe(Server):
    def __init__(self, dataset, algorithm, model, num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                 test_epochs, num_round, eval_gap, lamda):
        
        super().__init__(dataset, algorithm, model[0], num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                         num_round)
        self.eval_gap = eval_gap
        total_test_examples = 0
        self.beta = outer_lr
        self.K = test_epochs
        self.personal_learning_rate = 0.01

        if dataset in ['office-home', 'office_caltech_10']:
            train_loaders, test_loaders, train_full_loaders, test_full_loaders = read_officehome_data(BATCH_SIZE=batch_size) if dataset == 'office-home' else read_office_caltech(BATCH_SIZE=batch_size)

            total_clients = len(train_loaders)

            for i in trange(total_clients, desc="Create client"):
                client = clientpFedMe(i, [train_loaders[i], train_full_loaders[i]], [test_loaders[i], test_full_loaders[i]], model, batch_size, inner_lr, outer_lr, local_epochs, test_epochs, lamda)

                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test
        else:
            data = read_data(dataset)
            total_clients = len(data[0])
            for i in range(total_clients):
                id, train , test = read_client_data(i, data, dataset)
                client = clientpFedMe(i, train, test, model, batch_size, inner_lr, outer_lr, local_epochs, test_epochs, lamda)
                self.clients.append(client)
                self.total_train_examples += client.num_train
                total_test_examples += client.num_test

        print("Finished creating pFedMe server, total clients: {}, total train samples: {}, total test samples: {}"
              .format(total_clients, self.total_train_examples, total_test_examples))

    def send_grads(self):
        assert (self.clients is not None and len(self.clients) > 0)
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad)
        for user in self.clients:
            user.set_grads(grads)

    def train(self):
        loss = []
        for glob_iter in range(self.num_round):
            # send all parameter for clients 
            self.send_parameters()
            if glob_iter % self.eval_gap == 0 or glob_iter == self.num_round - 1:
                print("-------------Round number: ",glob_iter, " -------------")
                self.evaluate_personalized_model()

            # Evaluate gloal model on user for each interation
            # if glob_iter % self.eval_gap == 0:
                # print("Evaluate global model")
                # print("")
                # self.evaluate()

            # do update for all clients not only selected clients
            for user in self.clients:
                user.train() #* user.train_samples
            
            # choose several clients to send back upated model to server
            # self.personalized_evaluate()
            self.selected_clients = self.select_clients(self.num_select_clients)

            # Evaluate gloal model on user for each interation
            #print("Evaluate persionalized model")
            #print("")
            
            #self.aggregate_parameters()
            self.persionalized_aggregate_parameters()

        #print(loss)
        self.save_results()
        self.save_model()

    def evaluate_personalized_model(self):
        stats = self.test_persionalized_model()  
        # stats_train = self.train_error_and_loss_persionalized_model()
        glob_acc = np.sum(stats[2])*1.0/np.sum(stats[1])
        # train_acc = np.sum(stats_train[2])*1.0/np.sum(stats_train[1])
        # train_loss = np.dot(stats_train[3], stats_train[1])*1.0/np.sum(stats_train[1])
        # train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1])
        client_acc = np.array(stats[2]) / np.array(stats[1])
        self.client_acc.append(client_acc)
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append([])
        self.rs_train_loss_per.append([])
        #print("stats_train[1]",stats_train[3][0])

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
        
        print("Average Personal Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", [])
        print("Average Personal Trainning Loss: ",[])

    def test_persionalized_model(self):
        '''tests self.latest_model on given clients
        '''
        num_samples = []
        tot_correct = []
        for c in self.clients:
            ct, ns = c.test_persionalized_model()
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
        ids = [c.cid for c in self.clients]

        return ids, num_samples, tot_correct

    def train_error_and_loss_persionalized_model(self):
        num_samples = []
        tot_correct = []
        losses = []
        for c in self.clients:
            ct, cl, ns = c.train_error_and_loss_persionalized_model() 
            tot_correct.append(ct*1.0)
            num_samples.append(ns)
            losses.append(cl*1.0)
        
        ids = [c.cid for c in self.clients]
        #groups = [c.group for c in self.clients]

        return ids, num_samples, tot_correct, losses

    def add_parameters(self, user, ratio):
        model = self.model.parameters()
        for server_param, user_param in zip(self.model.parameters(), user.get_parameters()):
            server_param.data = server_param.data + user_param.data.clone() * ratio

    def persionalized_aggregate_parameters(self):
        assert (self.clients is not None and len(self.clients) > 0)

        # store previous parameters
        previous_param = copy.deepcopy(list(self.model.parameters()))
        for param in self.model.parameters():
            param.data = torch.zeros_like(param.data)
        total_train = 0
        #if(self.num_clients = self.to)
        for user in self.selected_clients:
            total_train += user.num_train

        for user in self.selected_clients:
            self.add_parameters(user, user.num_train / total_train)
            #self.add_parameters(user, 1 / len(self.selected_clients))

        # aaggregate avergage model with previous model using parameter beta 
        for pre_param, param in zip(previous_param, self.model.parameters()):
            param.data = (1 - self.beta)*pre_param.data + self.beta*param.data

    def send_parameters(self):
        assert (self.clients is not None and len(self.clients) > 0)
        for user in self.clients:
            user.set_parameters(self.model)