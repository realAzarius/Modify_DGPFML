import copy

import torch.nn as nn

from clients.clientbase import Client
from utils.language_utils import *


class ClientIFCA(Client):
    def __init__(self, cid, train_data, test_data, model, recv_models, batch_size, inner_lr, local_epochs):
        super().__init__(cid, train_data, test_data, model, batch_size, inner_lr, local_epochs)

        # self.optimizer = MySGD(self.model.parameters(), lr=self.inner_lr)

        self.recv_models = [copy.deepcopy(model) for model in recv_models]
        self.model = recv_models[0]

        if 'dnn' in self.model_name:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.model_name = model[1]

    def set_k_models(self, k_models):
        self.recv_models = [copy.deepcopy(model) for model in k_models]

    def train(self):
        optimal_cluster_ID = 0
        min_loss = float('inf')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainX, trainY = [], []
        for epoch in range(self.local_epochs):
            X, y = self.gen_next_train_batch()
            X, y = X.to(device), y.to(device)
            trainX.append(X)
            trainY.append(y)

        for cid, model in enumerate(self.recv_models):
            model.train()
            avg_loss = 0.0
            model.to(device)

            for epoch in range(self.local_epochs):
                X, y = trainX[epoch], trainY[epoch]

                # self.optimizer.zero_grad()
                output = model(X)
                loss = self.loss(output, y)

                grads = torch.autograd.grad(loss, model.parameters())
                avg_loss += loss

                for gd, params in zip(grads, model.parameters()):
                    params.data.add_(-self.inner_lr * gd)

            avg_loss /= self.local_epochs
            if avg_loss < min_loss:
                optimal_cluster_ID = cid
            model.to('cpu')

        # update self model
        self.model = self.recv_models[optimal_cluster_ID]
        return optimal_cluster_ID
