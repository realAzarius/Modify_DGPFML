import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import read_data, read_client_data
from torch.utils.data import DataLoader

num_rounds = 699
total_clients = 100
dataset = "cifia100"
data = read_data(dataset)

class EmbModel(nn.Module):

    def __init__(self, input_size, embedding_size):
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

def save_results(cid, epochs, emb):
    save_name = 'C' + str(cid) + '_E' + str(epochs) + '_result'

    # save embedding
    with open(save_name+'.txt', 'a+') as emb_fp:
        emb_fp.write(','.join(list(map(str, emb)))+'\n')

for c in range(total_clients):
    cid, train_data, test_data = read_client_data(c, data, dataset)
    model_path = os.path.join("saved_models", "cifia100", 'C'+cid+'_E'+str(num_rounds)+'_result.pt')
    # load model
    if os.path.exists(model_path):
        print('Loading model from ', model_path)
        model = torch.load(model_path)
        train_data_fullloader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)

        model.eval()
        with torch.no_grad():
            X, y = next( iter(train_data_fullloader) )
            X = X.reshape(-1, 3072)
            encoded = torch.zeros([1, 15], dtype=torch.float)
            encoded = model.encode(X)
        emb = torch.mean(encoded, axis=0).numpy().tolist()

        save_results(cid, num_rounds, emb)