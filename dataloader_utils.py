# dataloader_utils.py

import tenseal
from torch.utils.data import DataLoader, default_collate

import torch
from torch.utils.data.dataloader import default_collate
import tenseal as ts

def custom_collate(batch):
    # 将 CKKSVector 转换为列表或 numpy 数组
    converted_batch = []
    for item in batch:
        converted_item = [tensor.decrypt().tolist() if isinstance(tensor, ts.CKKSVector) else tensor for tensor in item]
        converted_batch.append(converted_item)
    return default_collate(converted_batch)

def get_data_loader(data, batch_size, collate_fn=None):
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)