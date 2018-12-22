import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

class Set_Data(Dataset):
    def __init__(self,data):
        self.data = data

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        X = np.array(self.data.iloc[index, 0:4].tolist(), dtype=np.float32)
        y = np.array(self.data.iloc[index, 4].tolist(), dtype=np.int64)
        return X, y
    
def get_data(file_path, purpose):
    data = pd.read_csv(file_path)
    data['class'] = data['class'].map({'Iris-setosa': 0, 
                                       'Iris-versicolor': 1, 
                                       'Iris-virginica': 2})
    
    train_data = data.sample(frac=0.8,random_state=99)
    test_data=data.drop(train_data.index)
    
    if purpose.lower() == "train":
        data = train_data
    else:
        data = test_data
    
    return Set_Data(data)