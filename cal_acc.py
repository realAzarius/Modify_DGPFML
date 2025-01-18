import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from torchvision import datasets, transforms
import numpy as np

# CIFAR-100数据集加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化处理，三个通道
])

# 下载训练集
trainset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 将图片数据转换为NumPy数组，以便于T-SNE处理
images = []
labels = []
for _, (img, label) in enumerate(trainset):
    images.append(img.view(3, 32, 32).numpy())  # 保持图片的三个通道
    labels.append(label)

images = np.array(images)
images = images.transpose((0, 2, 3, 1))  # 转换为(图片数量, 高度, 宽度, 通道)
images = images / 255.0  # 归一化到[0, 1]区间

# 展平图片为一维
images_flattened = images.reshape(images.shape[0], -1)

# DBSCAN聚类
dbscan = DBSCAN(eps=0.5, min_samples=10)  # eps和min_samples需要根据数据调整
dbscan.fit(images_flattened)
dbscan_labels = dbscan.labels_

# 执行T-SNE降维
tsne = TSNE(n_components=2, random_state=0)
dbscan_tsne_results = tsne.fit_transform(images_flattened[dbscan_labels != -1])  # 只对核心点进行T-SNE

# 绘制DBSCAN聚类的T-SNE散点图
plt.figure(figsize=(12, 6))
scatter = plt.scatter(dbscan_tsne_results[:, 0], dbscan_tsne_results[:, 1], c=dbscan_labels[dbscan_labels != -1], cmap='viridis', alpha=0.6, edgecolor='w')
plt.colorbar(scatter, label='DBSCAN Cluster Label')
plt.title('DBSCAN t-SNE visualization of CIFAR-100 dataset')
plt.show()

# KMeans聚类
num_clusters = 100  # CIFAR-100有100个类别
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
kmeans.fit(images_flattened)
cluster_centers = kmeans.cluster_centers_
kmeans_labels = kmeans.labels_

# 执行T-SNE降维
kmeans_tsne_results = tsne.fit_transform(cluster_centers)

# 为聚类中心创建标签数组
# 聚类中心的标签应该是原始数据点的标签的众数
# 确保labels是整数类型
labels = np.array(labels, dtype=np.int32)
cluster_labels = np.array([np.bincount(labels[(kmeans_labels == i)]).argmax() for i in range(num_clusters) if len(labels[(kmeans_labels == i)]) > 0])

# 绘制KMeans聚类的T-SNE散点图
plt.figure(figsize=(12, 6))
scatter = plt.scatter(kmeans_tsne_results[:, 0], kmeans_tsne_results[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, edgecolor='w')
plt.colorbar(scatter, label='Cluster Label')
plt.title('KMeans t-SNE visualization of CIFAR-100 dataset')
plt.show()