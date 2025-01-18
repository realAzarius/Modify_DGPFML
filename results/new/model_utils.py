import string

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset

from utils.language_utils import *
import os
import json
import torch
import numpy as np


IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNELS = 1

IMAGE_SIZE_CIFAR = 32
NUM_CHANNELS_CIFAR = 3

SHAKESPEARE_CONFIG = {
    "input_size": 80,
    "embed_size": 8,
    "hidden_size": 256,
    "output_size": 80,
    "n_layers": 2,
    "chunk_len": 80
}

def gen_next_batch(data, batch_size, num_iter):
    """Partition `data` into `data/batch_size` parts, sample the num_iter part.

    Args:
        data: data source
        batch_size: batch size
        num_iter: part to be returned

    Returns:
        Returns (X, y), which are both numpy array of length: batch_size
    """
    data_x = data['x']
    data_y = data['y']
    index = len(data_y)

    for i in range(num_iter):
        index += batch_size
        if index+batch_size > len(data_y):
            index = 0
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        batched_x = data_x[index: index+batch_size]
        batched_y = data_y[index: index+batch_size]

        yield batched_x, batched_y


def read_data(dataset):
    train_data_dir = os.path.join('data', dataset,'data','train')#'train'    'data','train'
    test_data_dir = os.path.join('data', dataset,'data','test')#'test'       'data','test'
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)

        clients.append(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        clients.append(cdata['users'])
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, train_data, test_data, groups


def read_client_data(index, data, dataset):
    cid = data[0][index]  # client name
    train_data = data[1][cid]
    test_data = data[2][cid]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if dataset == "mnist":
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif dataset == "cifar-100":
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif dataset == "shakespeare":
        train_data_res = [(process_x(x), process_y(y)) for x, y in zip(X_train, y_train)]
        test_data_res = [(process_x(x), process_y(y)) for x, y in zip(X_test, y_test)]
        return cid, train_data_res, test_data_res
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data_res = [(x, y) for x, y in zip(X_train, y_train)]
    test_data_res = [(x, y) for x, y in zip(X_test, y_test)]
    return cid, train_data_res, test_data_res

def get_number_of_params(net):
    return sum(x.numel() for x in net.parameters())

def get_number_of_trained_params(net):
    res = 0
    for params in net.parameters():
        if params.requires_grad:
            res += res.numel()
    return res

'''
def read_data(dataset):
    train_data_dir = os.path.join('data', dataset,'train')#'train'
    test_data_dir = os.path.join('data', dataset,'test')#'test'
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.append(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        clients.append(cdata['users'])
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    return clients, train_data, test_data, groups


def read_client_data(index, data, dataset):
    cid = data[0][index]  # client name
    train_data = data[1][cid]
    test_data = data[2][cid]
    X_train, y_train, X_test, y_test = train_data['x'], train_data['y'], test_data['x'], test_data['y']
    if dataset == "mnist":
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif dataset == "cifar-100":
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_CIFAR, IMAGE_SIZE_CIFAR).type(
            torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)
    elif dataset == "shakespeare":
        train_data_res = [(process_x(x), process_y(y)) for x, y in zip(X_train, y_train)]
        test_data_res = [(process_x(x), process_y(y)) for x, y in zip(X_test, y_test)]
        return cid, train_data_res, test_data_res
    else:
        X_train = torch.Tensor(X_train).type(torch.float32)
        y_train = torch.Tensor(y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).type(torch.float32)
        y_test = torch.Tensor(y_test).type(torch.int64)

    train_data_res = [(x, y) for x, y in zip(X_train, y_train)]
    test_data_res = [(x, y) for x, y in zip(X_test, y_test)]
'''
