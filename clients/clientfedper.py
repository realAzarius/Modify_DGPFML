import torch.nn as nn

from clients.clientbase import Client
from optimizers.fedoptimizer import MySGD
from utils.language_utils import *


class ClientFedPer(Client):
    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, local_epochs):
        super().__init__(cid, train_data, test_data, model, batch_size, inner_lr, local_epochs)

        self.optimizer = MySGD(self.model.parameters(), lr=self.inner_lr)

        # if model[1] == "Mclr_CrossEntropy" or model[1] == "lstm" or "init" in model[1]:
        self.loss = nn.CrossEntropyLoss()
        # else:
        #     self.loss = nn.NLLLoss()
        self.model_name = model[1]

    def train(self):
        self.model.train()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        #restore classifier parameters
        self.clone_model_params(self.local_model.features.parameters(), self.model.features.parameters())

        self.model.to(device)

        for epoch in range(self.local_epochs):
            X, y = self.gen_next_train_batch()

            X = X.to(device)
            y = y.to(device)

            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

        self.model.to("cpu")
        self.clone_model_params(self.model.parameters(), self.local_model.parameters())


    # FedAvg + Update
    def train_one_step(self):
        """
        test meta-model by one gradient descent
        """
        test_epochs = 1
        for epoch in range(test_epochs):  # local update
            self.model.train()
            # 1. update inner model
            X, y = next(iter(self.test_loader_full))

            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
        self.model.to("cpu")
