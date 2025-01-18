import copy
import os

import h5py
import numpy as np
import torch


class Server(object):

    def __init__(self, dataset, algorithm, model, num_select_clients, batch_size, inner_lr, outer_lr, local_epochs,
                 num_round):
        self.dataset = dataset
        self.algorithm = algorithm
        self.model = copy.deepcopy(model)
        self.num_select_clients = num_select_clients
        self.batch_size = batch_size
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.local_epochs = local_epochs
        self.num_round = num_round
        self.total_train_examples = 0
        self.total_test_examples = 0

        self.clients = []
        self.selected_clients = []

        self.client_acc, self.rs_train_acc, self.rs_train_loss, self.rs_glob_acc, self.rs_train_acc_per, self.rs_train_loss_per, self.rs_glob_acc_per = [], [], [], [], [], [], []

        # calculate the time spent on each round
        self.time_per_round = []        

    def add_params(self, client, ratio):
        for server_param, client_param in zip(self.model.parameters(), client.get_model_params()):
            server_param.data += client_param.data.clone() * ratio

    def aggregate_params(self):
        """Aggregate selected clients' model parameters"""
        # set server model parameters to zero
        for server_param in self.model.parameters():
            server_param.data = torch.zeros_like(server_param.data)
        total_train = 0
        for client in self.selected_clients:
            total_train += client.num_train
        for client in self.selected_clients:
            self.add_params(client, client.num_train / total_train)

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
        global stats_train
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
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1]) if self.dataset != "office-home" else []

        self.client_acc.append(client_acc)
        self.rs_glob_acc.append(global_acc)
        self.rs_train_acc.append(train_acc)
        self.rs_train_loss.append(train_loss)

        print("Average Global Testing Accurancy: ", global_acc)
        print("Average Global Trainning Accurancy: ", train_acc)
        print("Average Global Trainning Loss: ", train_loss)

    def evaluate_one_step(self):
        global stats_train
        if self.dataset == "office_caltech_10":
            test_loaders = []
            for c in self.clients:
                test_loaders.append(c.test_loader_full)
            
            for cid, c in enumerate(self.clients):
                cmodel = c.model
                for tid, test_loader in enumerate(test_loaders):
                    if self.algorithm == "pFedInit":
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
                    if self.algorithm == "pFedInit":
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
        train_acc = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[1]) if self.dataset not in ["office-home", 'office_caltech_10'] else []
        train_loss = sum([x * y for (x, y) in zip(stats_train[1], stats_train[3])]).item() / np.sum(stats_train[1]) if self.dataset not in ["office-home", 'office_caltech_10'] else []
        
        self.client_acc.append(client_acc)
        self.rs_glob_acc_per.append(glob_acc)
        self.rs_train_acc_per.append(train_acc)
        self.rs_train_loss_per.append(train_loss)

        print("Average Personal Testing Accurancy: ", glob_acc)
        print("Average Personal Trainning Accurancy: ", train_acc)
        print("Average Personal Trainning Loss: ", train_loss)

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
            alg += '_' + str(self.K) + 'k'

        if (len(self.rs_glob_acc) != 0 & len(self.rs_train_acc) & len(self.rs_train_loss)):
            with h5py.File("./results/" + '{}.h5'.format(alg), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_round_time', data=self.time_per_round)
                hf.create_dataset('rs_client_acc', data=self.client_acc)
                hf.close()

        if (len(self.rs_glob_acc_per) != 0 & len(self.rs_train_acc_per) & len(self.rs_train_loss_per)):
            with h5py.File("./results/" + '{}.h5'.format(alg), 'w') as hf:
                hf.create_dataset('rs_glob_acc', data=self.rs_glob_acc_per)
                hf.create_dataset('rs_train_acc', data=self.rs_train_acc_per)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss_per)
                hf.create_dataset('rs_round_time', data=self.time_per_round)
                hf.create_dataset('rs_client_acc', data=self.client_acc)
                hf.close()

    def save_model(self):
        model_path = os.path.join("saved_models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, self.algorithm + "_server" + ".pt"))

    def tSNEVisual(self, save_name, models):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        input_vector = []
        labels = []

        '''
        for c in self.clients:
            if self.algorithm == 'CFML':
                labels.append(c.train_one_step())
            else:
                c.train_one_step()
                if self.algorithm == 'Ours':
                    labels.append(self.client_belong_to_cluster[c.cid])
                else:
                    labels.append(c.cid)
        '''
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

    def flatten_model_parameters(self, parameters):
        return torch.cat([x.flatten() for x in parameters], 0).detach().numpy().tolist()
