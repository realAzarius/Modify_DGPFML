import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class OfficeDataset(Dataset):
    def __init__(self, base_path, site, train=True, transform=None):
        if train:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/raw/{}_train.pkl'.format(site), allow_pickle=True)
        else:
            self.paths, self.text_labels = np.load('./data/office_caltech_10/raw/{}_test.pkl'.format(site), allow_pickle=True)
            
        label_dict={'back_pack':0, 'bike':1, 'calculator':2, 'headphones':3, 'keyboard':4, 'laptop_computer':5, 'monitor':6, 'mouse':7, 'mug':8, 'projector':9}
        self.labels = [label_dict[text] for text in self.text_labels]
        self.transform = transform
        self.base_path = base_path if base_path is not None else './data'

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path)

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

def read_office_caltech(BATCH_SIZE=16):
    data_base_path = './data'
    transform_office = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
            transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
            transforms.Resize([256, 256]),            
            transforms.ToTensor(),
    ])
    # amazon
    amazon_trainset = OfficeDataset(data_base_path, 'amazon', transform=transform_office)
    amazon_testset = OfficeDataset(data_base_path, 'amazon', transform=transform_test, train=False)
    # caltech
    caltech_trainset = OfficeDataset(data_base_path, 'caltech', transform=transform_office)
    caltech_testset = OfficeDataset(data_base_path, 'caltech', transform=transform_test, train=False)
    # dslr
    dslr_trainset = OfficeDataset(data_base_path, 'dslr', transform=transform_office)
    dslr_testset = OfficeDataset(data_base_path, 'dslr', transform=transform_test, train=False)
    # webcam
    webcam_trainset = OfficeDataset(data_base_path, 'webcam', transform=transform_office)
    webcam_testset = OfficeDataset(data_base_path, 'webcam', transform=transform_test, train=False)

    amazon_train_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=BATCH_SIZE, shuffle=True)
    amazon_train_full_loader = torch.utils.data.DataLoader(amazon_trainset, batch_size=len(amazon_trainset), shuffle=True)
    amazon_test_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=BATCH_SIZE, shuffle=False)
    amazon_test_full_loader = torch.utils.data.DataLoader(amazon_testset, batch_size=len(amazon_testset), shuffle=False)

    caltech_train_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=BATCH_SIZE, shuffle=True)
    caltech_train_full_loader = torch.utils.data.DataLoader(caltech_trainset, batch_size=len(caltech_trainset), shuffle=True)
    caltech_test_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=BATCH_SIZE, shuffle=False)
    caltech_test_full_loader = torch.utils.data.DataLoader(caltech_testset, batch_size=len(caltech_testset), shuffle=False)

    dslr_train_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=BATCH_SIZE, shuffle=True)
    dslr_train_full_loader = torch.utils.data.DataLoader(dslr_trainset, batch_size=len(dslr_trainset), shuffle=True)
    dslr_test_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=BATCH_SIZE, shuffle=False)
    dslr_test_full_loader = torch.utils.data.DataLoader(dslr_testset, batch_size=len(dslr_testset), shuffle=False)

    webcam_train_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=BATCH_SIZE, shuffle=True)
    webcam_train_full_loader = torch.utils.data.DataLoader(webcam_trainset, batch_size=len(webcam_trainset), shuffle=True)
    webcam_test_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=BATCH_SIZE, shuffle=False)
    webcam_test_full_loader = torch.utils.data.DataLoader(webcam_testset, batch_size=len(webcam_testset), shuffle=False)

    return [amazon_train_loader, caltech_train_loader, dslr_train_loader, webcam_train_loader], [amazon_test_loader, caltech_test_loader, dslr_test_loader, webcam_test_loader], [amazon_train_full_loader, caltech_train_full_loader, dslr_train_full_loader, webcam_train_full_loader], [amazon_test_full_loader, caltech_test_full_loader, dslr_test_full_loader, webcam_test_full_loader]


