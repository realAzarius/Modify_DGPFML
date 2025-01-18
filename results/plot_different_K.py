import h5py
import matplotlib.pyplot as plt
import numpy as np

data_paths = [
    'synthetic_8_Ours_10_5_0.01_0.005_10c_600r_2k.h5',
    'synthetic_8_Ours_10_5_0.01_0.005_10c_600r_4k.h5',
    'synthetic_8_Ours_10_5_0.01_0.005_10c_600r_8k.h5',
    'synthetic_8_Ours_10_5_0.01_0.005_10c_600r_10k.h5',

    'femnist_Ours_10_5_0.003_0.001_10c_800r_2k.h5',
    'femnist_Ours_10_5_0.003_0.001_10c_800r_4k.h5',
    'femnist_Ours_10_5_0.003_0.001_10c_800r_8k.h5',
    'femnist_Ours_10_5_0.003_0.001_10c_800r_10k.h5',

    'shakespeare_Ours_16_5_0.8_0.3_10c_600r_2k.h5',
    'shakespeare_Ours_16_5_0.8_0.3_10c_600r_4k.h5',
    'shakespeare_Ours_16_5_0.8_0.3_10c_600r_8k.h5',
    'shakespeare_Ours_16_5_0.8_0.3_10c_600r_10k.h5',
]
plot_style = [
    '#252a34',
    '#e84545',
    '#903749',
    '#1e6f5c',

    '#252a34',
    '#e84545',
    '#903749',
    '#1e6f5c',

    '#252a34',
    '#e84545',
    '#903749',
    '#1e6f5c',
]
plot_label = [
    'G-FML(2)',
    'G-FML(4)',
    'G-FML(8)',
    'G-FML(10)',

    'G-FML(2)',
    'G-FML(4)',
    'G-FML(8)',
    'G-FML(10)',

    'G-FML(2)',
    'G-FML(4)',
    'G-FML(8)',
    'G-FML(10)',
    'FedAvg',
    'IFCA',
    'Per-FedAvg',
    'G-FML(4)',
    'FedAvg',
    'IFCA',
    'Per-FedAvg',
    'G-FML(4)',
    'FedAvg',
    'IFCA',
    'Per-FedAvg',
    'G-FML(4)',
]
plot_title = '200 clients, FEMNIST'

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4))

plot_len = 3
cnt_len = 0

boxplot = []
ax_list = [ax1, ax2, ax3]
test_acc_list = []
for path in data_paths:
    data = h5py.File(path, 'r')
    losses = data['rs_train_loss'][::]
    train_acc = data['rs_train_acc'][::]
    test_acc = data['rs_glob_acc'][::]
    times = data['rs_round_time']
    times = np.cumsum(times)[::]*10

    test_acc_list.append(test_acc)
    cnt_len += 1
    
    if cnt_len % 4 == 0:
        if cnt_len//4 == 1:
            ax_list[cnt_len//4-1].set_title('100 clients, Synthetic(8)')
        elif cnt_len//4 == 2:
            ax_list[cnt_len//4-1].set_title('200 clients, FEMNIST')
        else:
            ax_list[cnt_len//4-1].set_title('103 clients, Shakespeare')
        # if cnt_len//4-1 == 0:
        ax_list[cnt_len//4-1].set_ylabel("Test Accuracy")
        ax_list[cnt_len//4-1].boxplot(test_acc_list, labels=['G-FML(2)','G-FML(4)','G-FML(8)','G-FML(10)'], showfliers=False)
        test_acc_list = []


# xx = [i+1 for i in range(len(test_acc_list))]
# ax1.plot(xx, test_acc_list, '--^', color='#21209c', markersize=10., mfc="None", label=l)
# ax1.set_xticks([1, 2, 3, 4])
# ax1.set_xticklabels(['K=1', 'K=4', 'K=8', 'K=10'])


# cnt_len = 0

# for path, s, l in zip(data_paths, plot_style, plot_label):
#     data = h5py.File(path, 'r')
#     losses = data['rs_train_loss'][::]
#     train_acc = data['rs_train_acc'][::]
#     test_acc = data['rs_glob_acc'][::]
#     times = data['rs_round_time']
#     times = np.cumsum(times)[::]*10

#     cnt_len += 1
#     if cnt_len <= 4:
#         # ax1.set_title('100 clients, synthetic(1)')
#         # ax1.set_ylabel("Test Accuracy")
#         # # ax1.set_xlabel("Number of rounds")
#         # x = [i*10 for i in range(len(losses+1))]
#         # ax1.plot(x, test_acc, s, mfc="None", label=l)
#         # ax1.legend()

#         x = [i*10 for i in range(len(losses+1))]
#         ax4.set_ylabel("Training loss")
#         ax4.set_xlabel("Number of rounds")
#         ax4.plot(x, losses, s, mfc="None", label=l)
#         ax4.legend()
#     elif cnt_len > 4 and cnt_len <= 8:
#         # ax2.set_title('100 clients, synthetic(4)')
#         # x = [i*10 for i in range(len(losses+1))]
#         # ax2.plot(x, test_acc, s, mfc="None", label=l)
#         # ax2.legend()

#         x = [i*10 for i in range(len(losses+1))]
#         ax5.set_ylabel("Training loss")
#         ax5.set_xlabel("Number of rounds")
#         ax5.plot(x, losses, s, mfc="None", label=l)
#         # ax5.legend()
#     else:
#         # ax3.set_title('100 clients, synthetic(8)')
#         # x = [i*10 for i in range(len(losses+1))]
#         # ax3.plot(x, test_acc, s, mfc="None", label=l)
#         # ax3.legend()

#         x = [i*10 for i in range(len(losses+1))]
#         ax6.set_ylabel("Training loss")
#         ax6.set_xlabel("Number of rounds")
#         ax6.plot(x, losses, s, mfc="None", label=l)
#         # ax6.legend()

plt.savefig('./diff_k.pdf', dpi=600)