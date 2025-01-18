import copy
import os

import torch.nn as nn
import torch.optim

from clients.clientbase import Client
from models.task_embedding import Autoencoder
from optimizers.fedoptimizer import MySGD
from utils.language_utils import *


class ClientOurs(Client):
    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, outer_lr, epochs, test_epochs):
        super().__init__(cid, train_data, test_data,
                         model, batch_size, inner_lr, epochs)

        # if model[1] == "Mclr_CrossEntropy" or model[1] == "lstm":
        #     self.loss = nn.CrossEntropyLoss()
        # else:
        #     self.loss = nn.NLLLoss()
        self.loss = nn.CrossEntropyLoss()

        # meta step size
        self.outer_lr = outer_lr
        self.meta_optim = MySGD(self.model.parameters(), lr=self.inner_lr)

        self.test_epochs = test_epochs
        self.embed_model = Autoencoder(input_size=3072, embedding_size=10)
        self.model_name = model[1]

    def train(self):
        # assert self.embed_model == None

        if self.embed_model is None:
            self.embed_model = Autoencoder(input_size=784, embedding_size=64)  # 替换为适当的输入和嵌入维度

        self.embed_model = self.embed_model.to('cpu')
        self.embed_model.eval()
        emb = []

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)

        self.model.train()
        for epoch in range(self.local_epochs):  # local update
            base_model = copy.deepcopy(list(self.model.parameters()))

            X, y = self.gen_next_train_batch()
            X = X.to(device)
            y = y.to(device)

            self.meta_optim.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.meta_optim.step()

            # 2. update meta-model
            X, y = self.gen_next_train_batch()

            X = X.to(device)
            y = y.to(device)

            self.meta_optim.zero_grad()
            output_q = self.model(X)
            loss_q = self.loss(output_q, y)
            loss_q.backward()

            # set model parameters to the raw model parameters
            for old_p, new_p in zip(self.model.parameters(), base_model):
                old_p.data = new_p.data.clone()

            self.meta_optim.step(beta=self.outer_lr)

            # compute the embedding of `X`
            if epoch == self.local_epochs - 1:
                with torch.no_grad():
                    X = X.to("cpu")
                    emb.append(torch.mean(self.embed_model.encode(X.view(X.shape[0], -1)), axis=0, keepdims=True))
        self.model.to("cpu")
        # save meta model
        self.clone_model_params(self.model.parameters(),
                                self.local_model.parameters())

        return torch.mean(torch.cat(emb, axis=0), axis=0).numpy().reshape(1, -1)

    def get_embed_model(self):
        return self.embed_model

    def train_one_step(self):
        """
        test meta-model by one gradient descent
        """
        self.model.train()
        for epoch in range(self.test_epochs):  # local update
            # 1. update inner model
            X, y = next(iter(self.test_loader_full))
            self.meta_optim.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.meta_optim.step()

    def pre_trains_ae(self, pre_train_epochs, dataset):
        model_path = os.path.join(
            'saved_models', dataset, 'embed_' + str(self.cid) + '.pt')

        if os.path.exists(model_path):
            print('Loading encoder sucessfully!')
            self.embed_model = torch.load(model_path)
        else:
            print('Training encoder!')
            if self.model_name == "lstm":
                self.embed_model = Autoencoder(input_size=80, embedding_size=8)
            elif self.model_name == "mclr":
                self.embed_model = Autoencoder(input_size=60, embedding_size=6)
            else:
                self.embed_model = Autoencoder(input_size=3072, embedding_size=64)

            optimizer = torch.optim.Adam(self.embed_model.parameters(), lr=0.05)
            loss_fn = nn.MSELoss()
            for epoch in range(pre_train_epochs):
                loss = 0
                for X, _ in self.train_dataloader:
                    if self.model_name == "lstm":
                        X = process_x(X)
                        X = torch.from_numpy(X)
                        X = X.T
                        X = X.float()
                    else:
                        X = X.view(X.shape[0], -1)

                    out, code = self.embed_model(X)
                    optimizer.zero_grad()
                    train_loss = loss_fn(out, X)
                    train_loss.backward()
                    optimizer.step()
                    loss += train_loss.item()
        # self.save_embedding_model(self.embed_model, dataset)

    def get_client_embedding(self):
        X, y = next(iter(self.train_loader_full))
        X = X.view(X.shape[0], -1)
        model = self.embed_model.to('cpu')
        model.eval()
        # print(X.shape)  # [43,3072]
        with torch.no_grad():
            encoded = model.encode(X)
        # print(encoded.shape)  # [43,10]
        return torch.mean(encoded, dim=0).numpy().tolist()

    def save_embedding_model(self, dataset):
        model_path = os.path.join('saved_models', dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.save(self.embed_model, os.path.join(
            model_path, 'embed_' + str(self.cid) + '.pt'))
