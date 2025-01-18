# @Time : 2024/11/18 14:19
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import torch
import numpy as np

# CIFAR-100数据集加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5,), (0.5,))  # 归一化处理
])

# 下载训练集
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 将图片数据转换为NumPy数组，以便于T-SNE处理
images = []
labels = []
for _, (img, label) in enumerate(trainset):
    images.append(img.view(-1).numpy())  # 将图片展平
    labels.append(label)

images = np.array(images)
images = images / 255.0  # 归一化到[0, 1]区间

# 动态聚类
num_clusters = 100  # CIFAR-100有100个类别
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(images)  # 对整个数据集进行聚类

# 选择聚类中心进行T-SNE可视化
cluster_centers = kmeans.cluster_centers_

# 执行T-SNE降维
tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(cluster_centers)

# 为聚类中心创建标签数组
# 由于聚类中心的标签与数据点的标签是一致的，我们可以直接使用kmeans.labels_
# 但是我们需要确保聚类中心的标签与CIFAR-100数据集中的真实标签一致
# 这里我们假设聚类中心的标签与数据点的标签是一致的
cluster_labels = kmeans.labels_

# 绘制T-SNE聚类图
plt.figure(figsize=(12, 12))
scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolor='w')
plt.colorbar(scatter, label='Cluster Label')
plt.title('t-SNE visualization of CIFAR-100 dataset after dynamic clustering')
plt.savefig('./DGPFML/2t_sne_cifar100.png')  # 保存到特定路径
plt.show()