import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import copy
import torch.nn as nn

from clients.clientbase import Client
from optimizers.fedoptimizer import MySGD
from utils.language_utils import *

class ClientPerAvg(Client):

    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, outer_lr, epochs, test_epochs):
        super().__init__(cid, train_data, test_data, model, batch_size, inner_lr, epochs)

        if 'dnn' in self.model_name:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        # meta step size
        self.outer_lr = outer_lr
        self.meta_optim = MySGD(self.model.parameters(), lr=inner_lr)

        self.test_epochs = test_epochs

    def train(self):

        self.model.train()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)

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

        # save meta model
        self.model.to("cpu")
        self.clone_model_params(self.model.parameters(), self.local_model.parameters())

    def train_one_step(self):
        """
        test meta-model by one gradient descent
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model.to(device)

        for epoch in range(self.test_epochs):  # local update 
            self.model.train()
            # 1. update inner model
            self.meta_optim.zero_grad()
            # Gradient accumulation
            X, y = next(iter(self.test_loader_full))
            X = X.to(device)
            y = y.to(device)
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.meta_optim.step()

        self.model.to('cpu')