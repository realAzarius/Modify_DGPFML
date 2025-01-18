import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import copy

from clients.clientbase import Client
from optimizers.fedoptimizer import MySGD
from utils.language_utils import *

class ClientpFedInit(Client):
    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, outer_lr, epochs, test_epochs, fixed_weight, E):
        super().__init__(cid, train_data, test_data,
                         model, batch_size, inner_lr, epochs)
        self.E = E
        if 'dnn' in self.model_name:
            self.loss = nn.NLLLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.outer_lr = outer_lr

        self.fixed_weight = fixed_weight

        if self.fixed_weight:
            self.load_pretrain_model()
            self.local_model = copy.deepcopy(self.model)

        params = []
        for k, v in self.model.named_parameters():
            if 'mtl' in k or 'classifier' in k:
                params.append(v)
        self.meta_optim = MySGD(params, lr=outer_lr)

        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.meta_optim, step_size=3, gamma=0.5)
        self.test_epochs = test_epochs

    def train(self, global_comm):
        self.model.train()
        # restore classifier parameters
        # self.clone_model_params(self.local_model.classifier.parameters(), self.model.classifier.parameters())
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(device)

        # 1. perform local update
        for epoch in range(self.local_epochs):  # local update
            # 1. update the local model, fixed feature weights
            if not self.fixed_weight: # update
                self.freeze_meta_parameters(True) # update features, classifier, freeze meta
            else:
                self.freeze_all_parameters(True) # freeze features and meta
            X, y = self.gen_next_train_batch()
            X = X.to(device)
            y = y.to(device)
            output = self.model(X)
            loss = self.loss(output, y)
            grad = torch.autograd.grad(loss, filter(lambda p: p.requires_grad, self.model.parameters()))
            for param, g in zip(filter(lambda p: p.requires_grad, self.model.parameters()), grad):
                param.data = param.data - self.inner_lr * g

            # 2. meta-update Scale and Shift, also the classifier
            self.freeze_meta_parameters(False)
            X, y = self.gen_next_train_batch()
            X = X.to(device)
            y = y.to(device)
            # self.meta_optim.zero_grad()
            output_q = self.model(X)
            loss_q = self.loss(output_q, y)
            # loss_q.backward()
            # self.meta_optim.step()
            grad = torch.autograd.grad(loss_q, filter(lambda p: p.requires_grad, self.model.parameters()))
            for param, g in zip(filter(lambda p: p.requires_grad, self.model.parameters()), grad):
                param.data = param.data - self.outer_lr * g
        
        # save meta model
        self.model.to("cpu")
        # if (global_comm+3) % self.E == 0:
        #     self.model.gen_new_feature_weights()
        self.clone_model_params(self.model.parameters(), self.local_model.parameters())
    
    def freeze_meta_parameters(self, flag):
        self.model.freeze_meta_parameters(mode=flag)
    
    def freeze_all_parameters(self, flag):
        self.model.freeze_all_parameters(mode=flag)
    
    def load_pretrain_model(self):
        # model_name = 'pre_' + self.model_name + '.pth'
        model_name = 'pre_lenet_mnist_fashion_init.pth'
        # model_name = 'pre_alexnet_init.pth'
        # model_name = 'FedAvg_server.pt'
        # model_name = 'pre_lenet_cifar_init.pth'
        # model_name = 'pre_resnet_init.pth'
        path = os.path.join('saved_models', 'pretrain', 'model', model_name)
        # model_name = 'PerAvg_server0.pt'
        # model_name = 'pFedMe_server.pt'
        # path = os.path.join('saved_models', 'office_caltech_10', model_name)
        pretrained_model = torch.load(path, map_location=lambda storage, loc: storage)
        pretrained_model_list = []
        for name, param in pretrained_model.items():
            if 'running' in name:
                continue
            pretrained_model_list.append(param)
        
        client_model_list = []
        for name, param in self.model.named_parameters():
            if 'mtl' in name:
                continue
            client_model_list.append(param)
        
        for cln, pre in zip(client_model_list, pretrained_model_list):
            cln.data = pre.data.clone()

        print(('*'*10) + '  Locally loads sucessfully  ' + ('*'*10))

    def train_one_step(self, test_loader_full=None):
        """
        test meta-model by one gradient descent
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        test_loader_full = self.test_loader_full if test_loader_full == None else test_loader_full

        self.model.to(device)

        for epoch in range(self.test_epochs):  # local update 
            self.model.train()
            # 1. update inner model
            self.freeze_meta_parameters(False)

            # Gradient accumulation
            # for X, y in self.test_loader_full:
            X, y = next(iter(test_loader_full))
            X = X.to(device)
            y = y.to(device)
            self.meta_optim.zero_grad()
            output = self.model(X)
            loss_q = self.loss(output, y)
            loss_q.backward()
            self.meta_optim.step(beta=self.inner_lr)

        self.model.to('cpu')
