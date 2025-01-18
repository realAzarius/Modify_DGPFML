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
    
    '#252a34',
    '#e84545',
    '#903749',
    
    '#252a34',
    '#e84545',
    '#903749',

    # '#1e6f5c',
]
plot_label = [
    'RCA',
    'FCA',
    'Ours',

    'RCA',
    'FCA',
    'Ours',

    'RCA',
    'FCA',
    'Ours',
]
plot_title = '200 clients, FEMNIST'

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
ax1.grid(axis='y')
ax2.grid(axis='y')
ax3.grid(axis='y')
cnt_len = 0

x = []

for path in data_paths:
    data = h5py.File(path, 'r')
    losses = data['rs_train_loss'][::]
    train_acc = data['rs_train_acc'][::]
    test_acc = data['rs_glob_acc'][::]
    times = data['rs_round_time']
    times = np.cumsum(times)[::]*10

    cnt_len += 1
    if cnt_len <= 4:
        x.append(test_acc)

        if cnt_len == 4:
            ax1.set_title('100 clients, Synthetic(8)', fontsize=15)
            ax1.set_ylabel("Test Accuracy", fontsize=15)
            ax1.set_xlabel("The number of groups", fontsize=15)
            ax1.boxplot(x, showfliers=False, patch_artist=True, labels=['2', '4', '8', '10'])
            x = []

    elif cnt_len >= 5 and cnt_len <= 8:
        x.append(test_acc)

        if cnt_len == 8:
            ax2.set_title('200 clients, FEMNIST', fontsize=15)
            ax2.set_ylabel("Test Accuracy", fontsize=15)
            ax2.set_xlabel("The number of groups", fontsize=15)
            ax2.boxplot(x, showfliers=False, patch_artist=True, labels=['2', '4', '8', '10'])
            x = []
        
    else:
        x.append(test_acc)

        if cnt_len == 12:
            ax3.set_title('103 clients, Shakespeare', fontsize=15)
            ax3.set_ylabel("Test Accuracy", fontsize=15)
            ax3.set_xlabel("The number of groups", fontsize=15)
            ax3.boxplot(x, showfliers=False, patch_artist=True, labels=['2', '4', '8', '10'])
            x = []

plt.savefig('./boxplot_diff_k.pdf', dpi=600, bbox_inches='tight')