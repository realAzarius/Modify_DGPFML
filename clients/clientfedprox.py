import torch.nn as nn

import copy
from clients.clientbase import Client
from optimizers.fedoptimizer import PerturbedGradientDescent
from utils.language_utils import *


class ClientFedProx(Client):
    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, local_epochs, glob_iter, lamda):
        super().__init__(cid, train_data, test_data, model, batch_size, inner_lr, local_epochs)

        self.glob_iter = glob_iter

        if 'dnn' in self.model_name:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self.model_name = model[1]
        self.global_params = copy.deepcopy(list(self.model.parameters()))

        self.optimizer = PerturbedGradientDescent(self.model.parameters(), lr=self.inner_lr, mu=lamda)

    def train(self, lr_decay=False):
        self.model.train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # cache global model initialized value to local model
        self.model.to(device)
        # self.clone_model_params(self.model.parameters(), self.local_model.parameters())

        for epoch in range(self.local_epochs):
            self.model.train()
            X, y = self.gen_next_train_batch()

            X = X.to(device)
            y = y.to(device)

            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step(self.global_params, device)

        self.model.to("cpu")

    def set_parameters(self, model):
        for new_param, global_param, param in zip(model.parameters(), self.global_params, self.model.parameters()):
            global_param.data = new_param.data.clone()
            param.data = new_param.data.clone()