import torch
import torch.nn as nn
import numpy as np
import copy
from optimizers.fedoptimizer import MySGD
from clients.clientbase import Client


class ClientCFML(Client):

    def __init__(self, cid, train_data, test_data, model, num_k, batch_size, inner_lr, outer_lr, epochs, test_epochs):
        super().__init__(cid, train_data, test_data, model[0], batch_size, inner_lr, epochs)

        if (model[1] == "Mclr_CrossEntropy"):
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = nn.NLLLoss()

        # meta step size
        self.outer_lr = outer_lr
        self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=self.outer_lr)

        self.K = num_k
        self.optimal_k = -1
        self.k_meta_models = [copy.deepcopy(model[0]) for i in range(self.K)]

        self.test_epochs = test_epochs

    def train(self):
        """
        1. get optimal cluster
        2. set self.model = optimal meta model
        3. train
        """
        X, y = self.gen_next_train_batch()
        losses = [0] * self.K

        for idx, meta_model in enumerate(self.k_meta_models):
            loss_val = self.calculate_loss(meta_model, self.loss, X, y)
            losses[idx] = loss_val

        # 1. calculate optimal cluster
        self.optimal_k = np.argmin(losses)

        # 2. set self.model = k_meta_models[optimal_k]
        self.set_model_params(self.k_meta_models[self.optimal_k])

        self.model.train()
        use_first_batch = True
        for epoch in range(self.local_epochs):  # local update
            # 1. update inner model
            if use_first_batch:
                use_first_batch = False
            else:
                X, y = self.gen_next_train_batch()

            output = self.model(X, vars=None)
            loss = self.loss(output, y)
            grad = torch.autograd.grad(loss, self.model.parameters())
            fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, self.model.parameters())))

            # 2. update meta-model
            X, y = self.gen_next_train_batch()

            self.meta_optim.zero_grad()
            output_q = self.model(X, fast_weights)
            loss_q = self.loss(output_q, y)
            loss_q.backward()
            self.meta_optim.step()

        # save meta model
        self.clone_model_params(self.model.parameters(), self.local_model.parameters())

        return self.optimal_k

    def set_k_model_params(self, k_meta_models):
        for k in range(self.K):
            self.clone_model_params(k_meta_models[k].parameters(), self.k_meta_models[k].parameters())

    def calculate_loss(self, model, criterion, X, y):
        model.eval()
        # run one gradient descent
        y_target = model(X)
        loss = criterion(y_target, y)

        grad = torch.autograd.grad(loss, model.parameters())
        fast_weights = list(map(lambda p: p[1] - self.inner_lr * p[0], zip(grad, model.parameters())))

        with torch.no_grad():
            # test model performance
            output = model(X, fast_weights)
            loss = criterion(output, y)
            loss_value = loss.item()

        return loss_value

    def train_one_step(self):
        """
        test meta-model by one gradient descent
        """
        use_first_batch = True
        X, y = self.gen_next_test_batch()
        # find optimal k
        losses = [0] * self.K
        for idx, meta_model in enumerate(self.k_meta_models):
            loss_val = self.calculate_loss(meta_model, self.loss, X, y)
            losses[idx] = loss_val
        # 1. calculate optimal cluster
        self.optimal_k = np.argmin(losses)
        # 2. set self.model to k_meta_model[optimal_k]
        self.set_model_params(self.k_meta_models[self.optimal_k])

        self.model.train()
        for epoch in range(self.test_epochs):  # local update
            # 1. update inner model
            if use_first_batch:
                use_first_batch = False
            else:
                X, y = self.gen_next_test_batch()

            output = self.model(X)
            loss = self.loss(output, y)
            grad = torch.autograd.grad(loss, self.model.parameters())
            for gd, params in zip(grad, self.model.parameters()):
                params.data.add_(-self.inner_lr * gd)

        return self.optimal_k