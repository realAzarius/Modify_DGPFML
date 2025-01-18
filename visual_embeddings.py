# 要添加一个新单元，输入 '# %%'
# 要添加一个新的标记单元，输入 '# %% [markdown]'
# %%
from utils.model_utils import read_data, read_client_data
import torch
import os
import torch.nn as nn
import multiprocessing
from torch.utils.data import DataLoader
import torch.nn.functional as F
from multiprocessing import Process
import tqdm

# %% [markdown]
# 定义 embedding 模型

# %%
class EmbModel(nn.Module):

    def __init__(self, input_size=3*32*32, embedding_size=32):
        super().__init__()
        self.embedding_size = embedding_size
        
        # Encoder specification
        self.enc_linear_1 = nn.Linear(input_size, self.embedding_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.embedding_size, input_size)

    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, code):
        code = self.enc_linear_1(code)
        return code
    
    def decode(self, code):
        out = F.sigmoid(self.dec_linear_1(code))
        return out

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]


        self.FC1 = nn.Linear(3*32*32, 256)
        self.FC2 = nn.Linear(256, 128)
        self.decFC1 = nn.Linear(128, 256)
        self.decFC2 = nn.Linear(256, 3*32*32)


    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded
    
    def encode(self, x):
        x = F.relu(self.FC1(x))
        x = self.FC2(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.decFC1(F.relu(x)))
        x = F.sigmoid(self.decFC2(x))
        return x


# %% [markdown]
# 定义客户端，继承自 Process 类

# %%
def save_results(cid, epochs, model, emb):
    save_name = 'C' + str(cid) + '_E' + str(epochs) + '_result'

    # save embedding
    with open(save_name+'.txt', 'a+') as emb_fp:
        emb_fp.write(','.join(list(map(str, emb)))+'\n')

    # save model
    model_path = os.path.join("saved_models", "cifia100")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(model, os.path.join(model_path, save_name + '.pt'))
    

'''
返回 embedding
'''
def train(cid, train_data, epochs, lr):
    batch_size = 128
    model_path = os.path.join("saved_models", "cifia100", 'C'+str(cid)+'_E100_result.pt')
    loss_fn = nn.BCELoss().cuda()
    
    # load model
    if os.path.exists(model_path):
        print('Loading model from ', model_path)
        model = torch.load(model_path)
    else:
        model = EmbModel()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    train_data_fullloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    for epoch in tqdm.tqdm(range(epochs)):
        model.to(device)
        for i, (X, _) in enumerate(train_data_loader, 0):
            X = X.reshape(-1, 3072)
            X = X.to(device)
            out, code = model(X)
            optimizer.zero_grad()
            train_loss = loss_fn(out, X)
            train_loss.backward()
            optimizer.step()
        
        if (epoch+1) % 500 == 0:
            model.eval()
            with torch.no_grad():
                X, y = next( iter(train_data_fullloader) )
                X = X.reshape(-1, 3072)
                X = X.to(device)
                encoded = model.encode(X)
            encoded = encoded.to("cpu")
            emb = torch.mean(encoded, axis=0).numpy().tolist()
            
            model.to("cpu")
            save_results(cid, epoch+1, model, emb)
    
    # return embedding
    # model.eval()
    # with torch.no_grad():
    #     encoded = torch.zeros([1, 15], dtype=torch.float)
    #     encoded = model.encode(X)
    # return torch.mean(encoded, axis = 0, keepdims = True).numpy().tolist()
    return torch.zeros((1, 2))


# %%
def tSNEVisual(save_name, input_vector):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        labels = []
        col = 0

        for i in range(100):
            labels.append(col)
            # next label
            if (i+1) % 5 == 0:
                col += 1

        # Scaling the coordinates to [0, 1]
        def plot_embedding(data):
            x_min, x_max = np.min(data, 0), np.max(data, 0)
            data = (data - x_min) / (x_max - x_min)
            return data
        
        tsne = TSNE(n_components=2, init='pca', random_state=0, n_jobs=30, verbose=1, n_iter=10000)
        X_tsne = tsne.fit_transform(input_vector)
        aim_data = plot_embedding(X_tsne)

        plt.figure()
        plt.subplot(111)
        plt.scatter(aim_data[:, 0], aim_data[:, 1], c=labels)
        plt.savefig(save_name, dpi=600)

# %% [markdown]
# 训练阶段

# %%
# 创建进程池
pool = multiprocessing.Pool(8)
result = []
embeddings = []

total_clients = 100
epochs = 5000
learning_rate = 0.05
dataset = "cifia100"
data = read_data(dataset)

for c in range(total_clients):
    print('current: ', c)
    cid, train_data, test_data = read_client_data(c, data, dataset)
    # train(c+1, train_data, epochs, learning_rate)
    result.append(pool.apply_async(train, args=(c+1, train_data, epochs, learning_rate, )))

pool.close()
pool.join()

# sudo sh -c "ulimit -n 65535 && exec su $LOGNAME"
# 77591

for res in result:
    embeddings.append(res.get())

tSNEVisual('cifia.pdf', embeddings)