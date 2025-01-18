import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
import numpy as np
file1_path = '/home/hlj/代码/GFML/results/femnist_Ours_20_10_0.003_0.008_10c__300r_.h5'

with h5py.File(file1_path, 'r') as f:
    dataset_names = list(f.keys())
    print(dataset_names)
    # 读取.h5文件
    k_values = f['random_k2'][:]
    print(k_values)

rounds = range(1, len(k_values) + 1)
fixed_k = [4] * len(rounds)

# 使用样条插值生成平滑曲线
#spl = make_interp_spline(rounds, k_values)
#smooth_rounds = np.linspace(min(rounds), max(rounds), 300)
#smoothed_k_values = spl(smooth_rounds)
# 设置Y轴刻度
#plt.yticks([0, 5, 10, 15, 10])
# 绘制平滑曲线
#plt.plot(smooth_rounds, smoothed_k_values, label='Dynamic K')

# 绘制折线和固定K值
plt.plot(rounds, k_values, label='Dynamic K')
plt.plot(rounds, fixed_k, label='Fixed K')

plt.xlabel('Number of rounds')
plt.ylabel('K value')
plt.legend(loc='upper left')
#plt.title('K value comparison')
plt.show()

# 保存图表为PDF文件
plt.savefig('/home/hlj/代码/GFML/results/K.comparison.pdf', dpi=600, bbox_inches='tight')