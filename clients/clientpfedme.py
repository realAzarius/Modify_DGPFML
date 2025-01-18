import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import time
import copy
import torch
import torch.nn as nn
from optimizers.fedoptimizer import pFedMeOptimizer
from clients.clientbase import Client


class clientpFedMe(Client):
    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, outer_lr, epochs, test_epochs, lamda):
        super().__init__(cid, train_data, test_data,
                         model, batch_size, inner_lr, epochs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lamda = lamda
        self.K = test_epochs
        self.learning_rate = inner_lr
        self.personal_learning_rate = 0.01

        if 'dnn' in self.model_name:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()


        self.optimizer = pFedMeOptimizer(self.model.parameters(), lr=self.personal_learning_rate, lamda=self.lamda)
        
        self.model.to(self.device)
        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model = copy.deepcopy(list(self.model.parameters()))
        self.persionalized_model_bar = copy.deepcopy(list(self.model.parameters()))

    def set_grads(self, new_grads):
        if isinstance(new_grads, nn.Parameter):
            for model_grad, new_grad in zip(self.model.parameters(), new_grads):
                model_grad.data = new_grad.data
        elif isinstance(new_grads, list):
            for idx, model_grad in enumerate(self.model.parameters()):
                model_grad.data = new_grads[idx]

    def to_device(self, obj):
        if torch.is_tensor(obj):
            return obj.to(self.device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.to_device(v)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.to_device(v))
            return res
        else:
            raise TypeError("Invalid type for move_to")

    def train(self):
        LOSS = 0
        self.model.train()
        self.model.to(self.device)
        for epoch in range(1, self.local_epochs + 1):  # local update
            
            self.model.train()
            X, y = self.gen_next_train_batch()
            X = X.to(self.device)
            y = y.to(self.device)

            
            # K = 30 # K is number of personalized steps
            for i in range(self.K):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.local_model = self.to_device(self.local_model)
                self.persionalized_model_bar, _ = self.optimizer.step(self.local_model)

            # update local weight after finding aproximate theta
            for new_param, localweight in zip(self.persionalized_model_bar, self.local_model):
                localweight.data = localweight.data - self.lamda* self.learning_rate * (localweight.data - new_param.data)

        self.model.to("cpu")
        #update local model as local_weight_upated
        #self.clone_model_paramenter(self.local_weight_updated, self.local_model)
        self.update_parameters(self.local_model)

        return LOSS

    def set_parameters(self, model):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def test_persionalized_model(self):
        self.model.eval()
        self.model.to(self.device)
        test_acc = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.test_loader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.model.to("cpu")
        self.update_parameters(self.local_model)
        return test_acc, y.shape[0]

    def train_error_and_loss_persionalized_model(self):
        self.model.eval()
        self.model.to(self.device)
        train_acc = 0
        loss = 0
        self.update_parameters(self.persionalized_model_bar)
        for x, y in self.train_loader_full:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
            #print(self.id + ", Train Accuracy:", train_acc)
            #print(self.id + ", Train Loss:", loss)
        self.model.to("cpu")
        self.update_parameters(self.local_model)
        return train_acc, loss , self.num_train

    def get_parameters(self):
        self.model.to("cpu")
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()