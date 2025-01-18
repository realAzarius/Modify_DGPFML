import os
import h5py
import matplotlib.pyplot as plt

# 定义多个.h5文件的路径和对应的标签
file_paths = [
    '/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.05_10c_600r_4k.h5',
    '/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.008_10c_600r_4k.h5',
    '/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.08_10c_600r_4k.h5',
    '/share/home/MP2209103/texts/results/new/synthetic4_Ours_10_5_0.01_0.005___10c_600r_4k.h5',
    '/share/home/MP2209103/texts/results/new/synthetic4_pFedInit_16_5_0.01_0.005_10c_600r.h5'
    #'/share/home/MP2209103/texts/results/new/femnist_Ours_20_10_0.003_0.008_10c_1200r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/femnist_Ours_20_10_0.005_0.003_10c_1200r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/mnist_Ours_20_20_0.005_0.005_20c_600r_1k.h5',
    #'/share/home/MP2209103/texts/results/new/mnist_Ours_20_10_0.003_0.008__10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic4_Ours_10_5_0.01_0.05__10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic4_Ours_10_5_0.01_0.005___10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.005_10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.08_10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.08_10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.005_10c_600r_4k.h5',

    # 添加更多的文件路径.../share/home/MP2209103/texts/results/new/
]
labels = ['K=2', 'K=4', 'K=6', 'DG-PFML']  # 你的标签列表

# 创建图表
fig, ax = plt.subplots()

# 遍历每个.h5文件
for file_path, label in zip(file_paths, labels):
    # 获取文件的数据
    data = h5py.File(file_path, 'r')
    #train_acc = data['rs_train_acc'][::]
    test_acc = data['rs_glob_acc'][::]
    #train_loss = data['rs_train_loss'][::]
    # 绘制数据，并设置标签
    ax.plot(test_acc, label=label)

    # 关闭文件
    data.close()

# 添加图例
ax.legend()

# 添加标题和轴标签
ax.set_xlabel('Number of rounds')
ax.set_ylabel('Test accurcy')

plt.rcParams['font.family'] = 'Arial Unicode MS'
# 保存图表为PDF文件
plt.savefig('/share/home/MP2209103/texts/results/new/K..pdf', dpi=600, bbox_inches='tight')

# 显示图表
plt.show()
'''
import os
import h5py
import matplotlib.pyplot as plt

# 定义多个.h5文件的路径和对应的标签
file_paths = [
    #'/home/hlj/代码/GFML/results/mnist_Ours_20_10_0.003_0.008__10c_600r_4k.h5',
    '/home/hlj/代码/GFML/results/mnist_Ours_10_20_0.003_0.008_10c_600r_.h5',
    '/home/hlj/代码/GFML/results/mnist_Ours_10_10_0.003_0.008_10c_600r.h5',
    '/home/hlj/代码/GFML/results/mnist_Ours_20_10_0.003_0.008_20c_600r.h5',
    '/home/hlj/代码/GFML/results/mnist_Ours_20_20_0.003_0.008_20c_600r.h5',
    #'/home/hlj/代码/GFML/results/mnist_Ours_20_20_0.008_0.008_20c_600r.h5',
    '/home/hlj/代码/GFML/results/mnist_Ours_20_30_0.003_0.008_20c_600r_.h5',
    # 添加更多的文件路径...
]
labels = ['Ours 1', 'Ours 2', 'Ours 3', 'Ours 4', 'Ours 5']

# 创建图表
fig, ax = plt.subplots()

# 遍历每个.h5文件
for file_path, label in zip(file_paths, labels):
    # 获取文件的数据
    data = h5py.File(file_path, 'r')
    train_acc = data['rs_train_acc'][::]
    test_acc = data['rs_glob_acc'][::]

    # 绘制数据，并设置标签
    ax.plot(test_acc, label=label)

    # 关闭文件
    data.close()

# 添加图例
ax.legend()

# 添加标题和轴标签
ax.set_title('Comparison under different parameter settings')
ax.set_xlabel('Number of rounds')
ax.set_ylabel('Test accuracy')

plt.rcParams['font.family'] = 'Arial Unicode MS'
# 保存图表为PDF文件
plt.savefig('/home/hlj/代码/GFML/results/accuracy.comparison13.pdf', dpi=600, bbox_inches='tight')

import os
import h5py
import matplotlib.pyplot as plt

# 定义多个.h5文件的路径和对应的标签
file_paths = [
    #'/share/home/MP2209103/texts/results/new/femnist_Ours_20_10_0.003_0.008_10c_1200r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/femnist_Ours_20_10_0.005_0.003_10c_1200r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/mnist_Ours_20_20_0.005_0.005_20c_600r_1k.h5',
    #'/share/home/MP2209103/texts/results/new/mnist_Ours_20_10_0.003_0.008__10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic4_Ours_10_5_0.01_0.05__10c_600r_4k.h5',
    #'/share/home/MP2209103/texts/results/new/synthetic4_Ours_10_5_0.01_0.005___10c_600r_4k.h5',
    '/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.005_10c_600r_4k.h5',
    '/share/home/MP2209103/texts/results/new/synthetic8_Ours_10_5_0.01_0.08_10c_600r_4k.h5',

    # 添加更多的文件路径.../share/home/MP2209103/texts/results/new/
]
labels = ['G-FML', 'PPPFL-CELDP']

# 创建图表
fig, ax = plt.subplots()

# 遍历每个.h5文件
for file_path, label in zip(file_paths, labels):
    # 获取文件的数据
    data = h5py.File(file_path, 'r')
    train_acc = data['rs_train_acc'][::]
    test_acc = data['rs_glob_acc'][::]

    # 绘制数据，并设置标签
    ax.plot(test_acc, label=label)

    # 关闭文件
    data.close()

# 添加图例
ax.legend(loc='lower right')

# 添加标题和轴标签
#ax.set_title('Comparison under different parameter settings')
ax.set_xlabel('Number of rounds')
ax.set_ylabel('Test accuracy')

plt.rcParams['font.family'] = 'Arial Unicode MS'
# 保存图表为PDF文件
plt.savefig('/share/home/MP2209103/texts/results/new/5.pdf', dpi=600, bbox_inches='tight')
'''