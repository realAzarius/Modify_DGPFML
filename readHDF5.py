# @Time : 2024/11/16 15:58
import h5py

# 替换为你的HDF5文件路径
file_path = './results/cifar-100-python_Ours_20_30_0.8_0.3_10c_600r_.h5'

# 使用with语句确保文件正确关闭
with h5py.File(file_path, 'r') as file:
    # 遍历文件中的所有数据集
    for dataset in file:
        # 打印数据集的名称和数据
        data = file[dataset][...]
        print(f"Dataset: {dataset}")
        print(f"Data: {data}")