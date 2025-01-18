from torchvision import datasets, transforms
import os
import torch
from torch.utils.data import DataLoader, random_split


root_dir = os.path.join('data', 'office-home', 'OfficeHome')

def countFiles(root_path):
    assert os.path.exists(root_path)
    total_files = 0
    item_list = os.listdir(root_path)
    if len(item_list) == 0:
        return 0
    for item in item_list:
        next_path = os.path.join(root_path, item)
        if os.path.isfile(next_path):
            total_files += 1
        else:
            total_files += countFiles(next_path)
    
    return total_files


def read_officehome_data(BATCH_SIZE=16):
    import json

    domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    num_domain_clients = 5
    domain_train = []
    domain_test = []

    for domain in domains:
        path = root_dir + '/' + domain
        num_samples = countFiles(path)
        num_train = int(num_samples * 0.8)
        num_test = num_samples - num_train

        # load images
        transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

        domain_samples = datasets.ImageFolder(path, transform=transform)

        train_set, test_set = random_split(domain_samples, [num_train, num_test])
        domain_train.append(train_set)
        domain_test.append(test_set)


    # Create data structure
    train_loaders = []
    test_loaders = []
    train_full_loaders, test_full_loaders = [], []

    
    for domain_id in range(len(domains)):
        num_train = len(domain_train[domain_id])
        num_each_client_samples = num_train // num_domain_clients

        train_loader = DataLoader(domain_train[domain_id], batch_size=1, shuffle=True)
        X_train = []
        for img in train_loader:
            img[0] = img[0].reshape(3, 224, 224)
            img[1] = torch.squeeze(img[1])
            X_train.append(img)
        
        test_loader = DataLoader(domain_test[domain_id], batch_size=1, shuffle=True)
        X_test = []
        for idx, img in enumerate(test_loader):
            if idx >= 100:
                break
            img[0] = img[0].reshape(3, 224, 224)
            img[1] = torch.squeeze(img[1])
            X_test.append(img)


        for i in range(num_domain_clients):
            st = i*num_each_client_samples
            ed = min((i+1)*num_each_client_samples, num_train)
            train_loaders.append(DataLoader(X_train[st:ed], batch_size=BATCH_SIZE))
            test_loaders.append(DataLoader(X_test, batch_size=BATCH_SIZE))

            train_full_loaders.append(DataLoader(X_train[st:ed], batch_size=len(X_train[st:ed])))
            test_full_loaders.append(DataLoader(X_test, batch_size=len(X_test)))

    print("Finish Generating Samples")

    return train_loaders, test_loaders, train_full_loaders, test_full_loaders



