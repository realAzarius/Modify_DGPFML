import copy
import random

from torch.utils.data import DataLoader

from utils.language_utils import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Client(object):
    """
    Base class for all local clients
    """

    def __init__(self, cid, train_data, test_data, model, batch_size, inner_lr, local_epochs):
        self.cid = cid

        self.model = copy.deepcopy(model[0])
        self.model_name = model[1]

        self.batch_size = batch_size
        self.inner_lr = inner_lr
        self.local_epochs = local_epochs

        self.train_data = train_data
        self.test_data = test_data

        if isinstance(train_data[0], DataLoader):
            self.train_dataloader = train_data[0]
            self.test_dataloader = test_data[0]
            self.train_loader_full = train_data[1]
            self.test_loader_full = test_data[1]

            self.num_train = len(train_data[1].dataset)
            self.num_test = len(test_data[1].dataset)
        else:
            self.num_train = len(train_data)
            self.num_test = len(test_data)
            setup_seed(0)
            self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            setup_seed(0)
            self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            setup_seed(0)
            self.train_loader_full = DataLoader(train_data, batch_size=self.num_train, shuffle=True)
            setup_seed(0)
            self.test_loader_full = DataLoader(test_data, batch_size=self.num_test, shuffle=True)

        self.iter_train = iter(self.train_dataloader)
        self.iter_test = iter(self.test_dataloader)

        self.local_model = copy.deepcopy(self.model)
        self.clone_model_params(self.model.parameters(), self.local_model.parameters())

    def get_model_params(self):
        """Get model parameters"""
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()

    def set_model_params(self, new_model):
        """Set model parameters"""
        for old_param, new_param, local_param in zip(self.model.parameters(), new_model.parameters(), self.local_model.parameters()):
            old_param.data = new_param.data.clone()
            local_param.data = new_param.data.clone()

    def gen_next_train_batch(self):
        """Generates next batch for training

        Returns:
            X, y: features, label
        """
        try:
            (X, y) = next(self.iter_train)
        except StopIteration:
            # self.train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
            self.iter_train = iter(self.train_dataloader)
            (X, y) = next(self.iter_train)
        return X, y

    def gen_next_test_batch(self):
        """Generates next batch for testing

        Returns:
            X, y: features, label
        """
        try:
            (X, y) = next(self.iter_test)
        except StopIteration:
            # self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=True)
            self.iter_test = iter(self.test_dataloader)
            (X, y) = next(self.iter_test)
        return X, y

    def clone_model_params(self, src_params, dest_params):
        """Clones src_params to dest_params

        Args:
            src_params: parameters of source model
            dest_params: parameters of destination model

        Returns:
            parameters of destination model
        """
        for src_param, dest_param in zip(src_params, dest_params):
            dest_param.data = src_param.data.detach().clone()
        return dest_params

    def test(self):
        self.model.eval()
        self.model.to("cpu")
        test_acc = 0

        with torch.no_grad():
            for X, y in self.test_loader_full:
                output = self.model(X)
                if self.model_name == "lstm":
                    predict = torch.max(output[:, -1, :], 1)[1].data.numpy()
                    y = torch.max(y, 1)[1].data.numpy()
                    test_acc += (predict == y).astype(int).sum()
                else:
                    test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()

        return test_acc, self.num_test

    def train_error_and_loss(self):
        """Get training error and loss

        Returns:
            train_acc: training accuracy
            loss: loss
            num_train: size of training set
        """

        self.model.eval()
        train_acc = 0
        loss = 0

        with torch.no_grad():
            for X, y in self.train_loader_full:
                output = self.model(X)
                if self.model_name == "lstm":
                    predict = torch.max(output[:, -1, :], 1)[1].data.numpy()
                    label = torch.max(y, 1)[1].data.numpy()
                    train_acc += (predict == label).astype(int).sum()
                    loss += self.loss(output, y)
                else:
                    train_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                    # y = y.view(len(y), -1)
                    loss += self.loss(output, y)

        return train_acc, loss, self.num_train

    def update_parameters(self, new_params):
        for param, new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_train_loader_full(self):
        return self.train_loader_full
