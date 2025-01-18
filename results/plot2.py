import h5py
import matplotlib.pyplot as plt
import numpy as np

data_paths = [
    'synthetic_Ours_10_5_0.01_0.005_10c_200r_4k_random.h5',
    'synthetic_Ours_10_5_0.01_0.005_10c_200r_4k_fixed.h5',
    'synthetic_Ours_10_5_0.01_0.005_10c_200r_4k_our.h5',
    
    'femnist_Ours_10_5_0.003_0.001_10c_800r_4k_random.h5',
    'femnist_Ours_10_5_0.003_0.001_10c_800r_4k_fixed.h5',
    'femnist_Ours_10_5_0.003_0.001_10c_800r_4k.h5',

    'shakespeare_Ours_16_5_0.8_0.3_10c_200r_4k_random.h5',
    'shakespeare_Ours_16_5_0.8_0.3_10c_200r_4k_fix.h5',
    'shakespeare_Ours_16_5_0.8_0.3_10c_600r_4k.h5'



    # 'synthetic1_FedAvg_10_5_0.01_0.3_10c_600r.h5',
    # 'synthetic1_FedProx_10_5_0.01_0.3_10c_600r.h5',
    # 'synthetic1_FedAvgUpdate_10_5_0.01_0.008_10c_600r.h5',
    # 'synthetic_1_IFCA_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_1_PerAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_1_Ours_10_5_0.01_0.005_10c_600r_4k.h5',

    # 'synthetic_4_FedAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic4_FedProx_10_5_0.01_0.008_10c_600r.h5',
    # 'synthetic4_FedAvgUpdate_10_5_0.01_0.001_10c_600r.h5',
    # 'synthetic_4_IFCA_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_4_PerAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_4_Ours_10_5_0.01_0.005_10c_600r_4k.h5',

    
    # 'synthetic_8_FedAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic8_FedProx_10_5_0.01_0.008_10c_600r.h5',
    # 'synthetic8_FedAvgUpdate_10_5_0.01_0.008_10c_600r.h5',
    # 'synthetic_8_IFCA_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_8_PerAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_8_Ours_10_5_0.01_0.005_10c_600r_4k.h5', 

    # 'synthetic_4_FedAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic4_FedProx_10_5_0.01_0.008_10c_600r.h5',
    # 'synthetic4_FedAvgUpdate_10_5_0.01_0.001_10c_600r.h5',
    # 'synthetic_4_IFCA_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_4_PerAvg_10_5_0.01_0.005_10c_600r.h5',
    # 'synthetic_4_Ours_10_5_0.01_0.005_10c_600r_4k.h5',
    
    # 'femnist_FedAvg_10_5_0.003_0.001_10c_1200r.h5',
    # 'femnist_FedProx_10_5_0.003_0.001_10c_1200r.h5',
    # 'femnist_FedAvgUpdate_10_5_0.003_0.001_10c_1200r.h5',
    # 'femnist_IFCA_10_5_0.003_0.001_10c_1200r.h5',
    # 'femnist_PerAvg_10_5_0.003_0.001_10c_1200r.h5',
    # 'femnist_Ours_10_5_0.003_0.001_10c_1200r_4k.h5',

    # 'shakespeare_FedAvg_16_5_0.8_0.3_10c_600r.h5',
    # 'shakespeare_FedProx_16_5_0.8_0.1_10c_600r_mu0001.h5',
    # 'shakespeare_FedAvgUpdate_16_5_0.8_0.3_10c_600r.h5',
    # 'shakespeare_IFCA_16_5_0.8_0.3_10c_600r.h5',
    # 'shakespeare_PerAvg_16_5_0.8_0.3_10c_600r.h5',
    # 'shakespeare_Ours_16_5_0.8_0.3_10c_600r_4k.h5',

]
plot_style = [
    '#252a34',
    '#5438DC',
    '#1e6f5c',

    '#252a34',
    '#5438DC',
    '#1e6f5c',

    '#252a34',
    '#5438DC',
    '#1e6f5c',



    '#252a34',
    '#5438DC',
    '#F08913',
    '#e84545',
    '#903749',
    '#1e6f5c',
    
    '#252a34',
    '#5438DC',
    '#F08913',
    '#e84545',
    '#903749',
    '#1e6f5c',
    
    '#252a34',
    '#5438DC',
    '#F08913',
    '#e84545',
    '#903749',
    '#1e6f5c',
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


    'FedAvg',
    'FedProx',
    'FedAvg+Update',
    'IFCA',
    'Per-FedAvg',
    'G-FML(4)',

    'FedAvg',
    'FedProx',
    'FedAvg+Update',
    'IFCA',
    'Per-FedAvg',
    'G-FML(4)',

    'FedAvg',
    'FedProx',
    'FedAvg+Update',
    'IFCA',
    'Per-FedAvg',
    'G-FML(4)',
]

marker = [
    'o',
    '<',
    's',
    '*',
    '+',
    'D'
]
plot_title = '200 clients, FEMNIST'

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 9))
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()
ax6.grid()
cnt_len = 0

for path, s, l in zip(data_paths, plot_style, plot_label):
    data = h5py.File(path, 'r')
    losses = data['rs_train_loss'][::]
    train_acc = data['rs_train_acc'][::]
    test_acc = data['rs_glob_acc'][::]
    times = data['rs_round_time']
    times = np.cumsum(times)[::]*10
    
    if path == 'shakespeare_Ours_16_5_0.8_0.3_10c_600r_4k.h5':
        print('ok')
        losses = data['rs_train_loss'][:21]
        train_acc = data['rs_train_acc'][:21]
        test_acc = data['rs_glob_acc'][:21]
        times = data['rs_round_time']
        times = np.cumsum(times)[:21]*10

    cnt_len += 1
    
    if cnt_len <= 3:
        ax1.set_title('100 clients, Synthetic(8)', fontsize=15)
        ax1.set_ylabel("Test Accuracy", fontsize=15)
        # ax1.set_xlabel("Number of rounds", fontsize=15)
        x = [i*10 for i in range(len(losses+1))]
        ax1.plot(x, test_acc, s, label=l)
        ax1.legend()
        
        # ax4.set_title('200 clients, Synthetic(8)')
        ax4.set_ylabel("Training loss", fontsize=15)
        ax4.set_xlabel("Number of rounds", fontsize=15)
        ax4.plot(x, losses, s, label=l)
        ax4.legend()
    elif cnt_len >= 4 and cnt_len <= 6:
        ax2.set_title('200 clients, FEMNIST', fontsize=15)
        # ax2.set_ylabel("Test Accuracy")
        # ax2.set_xlabel("Number of rounds")
        x = [i*10 for i in range(len(losses+1))]
        ax2.plot(x, test_acc, s, label=l)
        ax2.legend()

        # ax5.set_title('100 clients, FEMNIST')
        # ax5.set_ylabel("Training loss", fontsize=15)
        ax5.set_xlabel("Number of rounds", fontsize=15)
        ax5.plot(x, losses, s, label=l)
        ax5.legend()
    else:
        ax3.set_title('103 clients, Shakespeare', fontsize=15)
        # ax3.set_ylabel("Test Accuracy")
        # ax3.set_xlabel("Number of rounds")
        x = [i*10 for i in range(len(losses+1))]
        ax3.plot(x, test_acc, s, label=l)
        ax3.legend()
        
        # ax6.set_title('103 clients, Shakespeare')
        # ax6.set_ylabel("Training loss", fontsize=15)
        ax6.set_xlabel("Number of rounds", fontsize=15)
        ax6.plot(x, losses, s, label=l)
        ax6.legend()

plt.savefig('./diff_grouping_strategies.pdf', dpi=600, bbox_inches='tight')
