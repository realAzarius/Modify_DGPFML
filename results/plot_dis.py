import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure


# load CFML data
cfml = h5py.File('./mnist_CFML_0.005_0.001_5c_200r_discrepancy.h5', 'r')
cfml_dis = cfml['rs_dis'][::]

cfml = h5py.File('./mnist_CFML_20_20_0.005_0.001_5c_200r_4k.h5', 'r')
cfml_losses = cfml['rs_train_loss'][::]
cfml_train_acc = cfml['rs_train_acc'][::]
cfml_test_acc = cfml['rs_glob_acc'][::]
cfml_times = cfml['rs_round_time']
cfml_times = np.cumsum(cfml_times)[::]


figure(num=None, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')

plt.title("1000 clients, synthetic data")
plt.xlabel("Number of rounds")


plt.plot(cfml_test_acc, '-', mfc="None", label='test acc')
plt.plot(cfml_dis[::10], '-', mfc="None", label='DIS')

plt.legend()

# plt.xlim(xmax=13)
plt.legend()
plt.savefig('./CFML_dis_mnist_4.png', dpi=600)