import h5py
import numpy as np

data_paths = [
    # 'cifar10(iid)_FedAvg_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(1)_FedAvg_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(3)_FedAvg_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(5)_FedAvg_64_10_0.01_0.005_5c_600r.h5',

    # 'cifar10(iid)_PerAvg_64_10_0.01_0.01_5c_600r.h5',
    # 'cifar10(1)_PerAvg_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(3)_PerAvg_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(5)_PerAvg_64_10_0.01_0.005_5c_600r.h5',

    # 'cifar10(iid)_pFedInit_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(1)_pFedInit_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(3)_pFedInit_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10(5)_pFedInit_64_10_0.01_0.005_5c_600r.h5',
    # 'cifar10_FedAvg_64_10_0.01_0.01_5c_600r.h5',
    # 'cifar10_FedPer_64_10_0.01_0.01_5c_600r.h5',
    'femnist_Ours_10_5_0.003_0.001_10c_1200r_4k.h5'

]

for i, path in enumerate(data_paths):
    data = h5py.File(path, 'r')
    test_acc = data['rs_glob_acc'][::]
    mina = np.min(test_acc)
    maxa = np.max(test_acc)
    mid = (mina+maxa)/2
    print('max: %.4f, diff: %.4f' % ( maxa, (maxa-mina)/2 ))
    # if (i+1) % 4 == 0:
    #     print('\n')