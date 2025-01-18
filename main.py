#!/usr/bin/env python
# !/home/hlj/anaconda3/envs/pytorch/lib/python3.8
import copy
import random
from configparser import ConfigParser
import json
import numpy as np
import torch
from torchvision.models import resnet18, alexnet
from models.models1 import DNN
from models.models1 import DNN_Init
from models.models1 import DNN_BN
from models.cnn import LeNet_Init, ResNetMtl
from models.cnn import Conv4_Init
from models.cnn import Conv4
from models.cnn import Conv4_BN
from models.cnn import LeNet
from models.cnn import LeNet_BN
from models.cnn import AlexNet_Init
from models.models1 import Mclr_Logistic
from models.models1 import NextCharacterLSTM
from servers.servercfml import ServerCFML
from servers.serverfedavg import ServerFedAvg
from servers.serverfedprox import ServerFedProx
from servers.serverifca import ServerIFCA
from servers.serverours import ServerOurs
from servers.serverfedinit import ServerpFedInit
from servers.serverperavg import ServerPerAvg
from servers.serverfedper import ServerFedPer
from servers.serverfedbn import ServerFedBN
from servers.serverpFedMe import serverpFedMe
from utils.model_utils import SHAKESPEARE_CONFIG


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def CNN(param):
    pass


def main(dataset, algorithm, model, batch_size, inner_lr, outer_lr, lamda, num_round, eval_gap, fixed_weight,
         local_epochs, test_epochs, optimizer, K_clusters, num_select_clients):
    # setup random seed
    setup_seed(999012)

    if model == "mclr":
        if dataset.startswith("synthetic"):
            model = Mclr_Logistic(60, 10), model
        else:
            model = Mclr_Logistic(60, 10), model
    elif model == "dnn":
        if dataset == "femnist":
            model = DNN(784, 100, 10), model
        else:
            model = DNN(784, 20, 10), model
    elif model == "dnn_bn":
        model = DNN_BN(784, 100, 62), model
    elif model == "LeNet_Init":
        if dataset == "cifar-100-python":
            model = LeNet_Init(in_channels=3, input_shape=(32, 32)), model
        else:
            model = LeNet_Init(in_channels=1, input_shape=(28, 28)), model
    elif model == "lenet":
        if dataset == "cifar-100":
            model = LeNet(3), model
        else:
            model = LeNet(), model
    elif model == "lenet_bn":
        model = LeNet_BN(3), model
    elif model == "alexnet":
        model = alexnet(pretrained=True), model
    elif model == "alexnet_init":
        model = AlexNet_Init(input_shape=(256, 256)), model
    elif model == "resnet_init":
        model = ResNetMtl(), model
    elif model == "resnet":
        model = resnet18(pretrained=True), model
    elif model == "conv4_init":
        model = Conv4_Init(in_channels=3, input_shape=(224, 224)), model
    elif model == "conv4_bn":
        model = Conv4_BN(in_channels=3, input_shape=(224, 224)), model
    elif model == "conv4":
        model = Conv4(in_channels=3, input_shape=(224, 224)), model
    elif model == "dnn_init":
        model = DNN_Init(784, 30, 10), model
    elif model == "lstm":
        # model = RNNModel('LSTM', 80, 8, 100, 1, 0.2, tie_weights=False), model
        model = NextCharacterLSTM(
            input_size=SHAKESPEARE_CONFIG["input_size"],
            embed_size=SHAKESPEARE_CONFIG["embed_size"],
            hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
            output_size=SHAKESPEARE_CONFIG["output_size"],
            n_layers=SHAKESPEARE_CONFIG["n_layers"]
        ), model

    if algorithm == "FedAvg" or algorithm == "FedAvgUpdate":
        is_update = False
        if algorithm == "FedAvgUpdate":
            is_update = True

        server = ServerFedAvg(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                              inner_lr=inner_lr,
                              outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, eval_gap=eval_gap,
                              num_select_clients=num_select_clients, is_update=is_update)
        server.train()

    if algorithm == "FedProx":
        server = ServerFedProx(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                               inner_lr=inner_lr,
                               outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, lamda=lamda,
                               eval_gap=eval_gap,
                               num_select_clients=num_select_clients)
        server.train()

    if algorithm == "PerAvg":
        server = ServerPerAvg(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                              inner_lr=inner_lr,
                              outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs,
                              test_epochs=test_epochs,
                              num_select_clients=num_select_clients, eval_gap=eval_gap)
        server.train()

    if algorithm == "FedPer":
        server = ServerFedPer(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                              inner_lr=inner_lr,
                              outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, eval_gap=eval_gap,
                              num_select_clients=num_select_clients)
        server.train()

    if algorithm == "FedBN":
        server = ServerFedBN(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                             inner_lr=inner_lr,
                             outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, eval_gap=eval_gap,
                             num_select_clients=num_select_clients)
        server.train()

    if algorithm == "pFedInit":
        server = ServerpFedInit(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                                inner_lr=inner_lr,
                                outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs,
                                test_epochs=test_epochs,
                                num_select_clients=num_select_clients, eval_gap=eval_gap, fixed_weight=fixed_weight)
        server.train()

    if algorithm == "pFedMe":
        server = serverpFedMe(dataset=dataset, algorithm=algorithm, model=model, batch_size=batch_size,
                              inner_lr=inner_lr,
                              outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs,
                              test_epochs=test_epochs,
                              num_select_clients=num_select_clients, eval_gap=eval_gap, lamda=lamda)
        server.train()
    if algorithm == "Ours":
        models = [copy.deepcopy(model[0]) for i in range(K_clusters)]

        models[0] = copy.deepcopy(model[0])
        server = ServerOurs(dataset=dataset, algorithm=algorithm, model=model, k_meta_models=models,
                            batch_size=batch_size, inner_lr=inner_lr,
                            outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, test_epochs=test_epochs,
                            eval_gap=eval_gap,
                            num_select_clients=num_select_clients, num_k=K_clusters)
        server.train()
    if algorithm == "IFCA":
        if model[1] == "mclr":
            models = [Mclr_Logistic(60, 10) if dataset.startswith("synthetic") else Mclr_Logistic(
                784, 10) for _ in range(K_clusters)]
        elif model[1] == "cnn":
            models = [CNN(10) for _ in range(K_clusters)]
        elif model[1] == "dnn":
            models = [DNN(784, 100, 10) if dataset.startswith("synthetic") else DNN() for _ in range(K_clusters)]
        elif model[1] == "lstm":
            models = [NextCharacterLSTM(
                input_size=SHAKESPEARE_CONFIG["input_size"],
                embed_size=SHAKESPEARE_CONFIG["embed_size"],
                hidden_size=SHAKESPEARE_CONFIG["hidden_size"],
                output_size=SHAKESPEARE_CONFIG["output_size"],
                n_layers=SHAKESPEARE_CONFIG["n_layers"])
                for i in range(K_clusters)]
        elif model[1] == "lenet":
            models = [LeNet(3) if dataset == "cifar-100" else LeNet()
                      for _ in range(K_clusters)]
        else:
            models = [alexnet(pretrained=True) for _ in range(K_clusters)]

        server = ServerIFCA(dataset=dataset, algorithm=algorithm, model=model, k_models=models, num_k=K_clusters,
                            batch_size=batch_size, inner_lr=inner_lr,
                            outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, test_epochs=test_epochs,
                            num_select_clients=num_select_clients, eval_gap=eval_gap)
        server.train()

    if algorithm == "CFML":
        models = []
        if model[1] == "mclr":
            models = [Mclr_Logistic() if dataset == "mnist" else Mclr_Logistic(
                60, 10) for i in range(K_clusters)]

        if model[1] == "cnn":
            models = [CNN(10) for i in range(K_clusters)]

        if model[1] == "dnn":
            models = [DNN(784, 100, 10) if dataset == "femnist" else DNN(
                60, 20, 10) if dataset.startswith("synthetic") else DNN() for i in range(K_clusters)]

        models[0] = copy.deepcopy(model[0])
        server = ServerCFML(dataset=dataset, algorithm=algorithm, model=model, k_meta_models=models,
                            batch_size=batch_size, inner_lr=inner_lr,
                            outer_lr=outer_lr, num_round=num_round, local_epochs=local_epochs, test_epochs=test_epochs,
                            eval_gap=eval_gap,
                            num_select_clients=num_select_clients, num_k=K_clusters)
        server.train()


if __name__ == '__main__':
    config = ConfigParser()
    config.read('config/config.ini', encoding='utf-8')

    config = config[config.sections()[0]]
    algorithm = config['algorithm']
    batch_size = config.getint('batch_size')
    inner_lr = config.getfloat('inner_lr')
    outer_lr = config.getfloat('outer_lr')
    test_epochs = config.getint('test_epochs')
    eval_gap = config.getint('eval_gap')
    optimizer = config['optimizer']
    K_clusters = config.getint('K_clusters')
    num_select_clients = config.getint('num_select_clients')
    num_round = config.getint('num_round')
    local_epochs = config.getint('local_epochs')
    dataset = config['dataset']
    model = config['model']
    lamda = config.getfloat('lamda')
    fixed_weight = config.getboolean('fixed_weight')

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm                               : {}".format(algorithm))
    print("Batch size                              : {}".format(batch_size))
    print("Learing rate                            : {}".format(inner_lr))
    print("Subset of selected clients per round    : {}".format(num_select_clients))
    print("Number of global rounds                 : {}".format(num_round))
    print("Number of local epochs                  : {}".format(local_epochs))
    print("Dataset                                 : {}".format(dataset))
    print("Local Model                             : {}".format(model))
    print("Fixed Weight                            : {}".format(fixed_weight))
    print("=" * 80)

    main(
        dataset=dataset,
        algorithm=algorithm,
        model=model,
        batch_size=batch_size,
        inner_lr=inner_lr,
        outer_lr=outer_lr,
        num_round=num_round,
        K_clusters=K_clusters,
        local_epochs=local_epochs,
        test_epochs=test_epochs,
        eval_gap=eval_gap,
        optimizer=optimizer,
        lamda=lamda,
        fixed_weight=fixed_weight,
        num_select_clients=num_select_clients,
    )
