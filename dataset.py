import numpy as np
import datetime
from torch.utils.data import DataLoader, Dataset
from untils.norm_utils import normalize_vector
import torch.nn.functional as F

class Triplet_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patent = self.df.iloc[idx]
        A = next(iter(patent.Anchor.values()))
        P = next(iter(patent.Positive.values()))
        N = next(iter(patent.Negative.values()))
        return np.array(A, dtype=np.float32), np.array(P, dtype=np.float32), np.array(N, dtype=np.float32)
    
class Test_Dataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patent = self.df.iloc[idx]
        encoded_feature = np.array(patent['encoded_feature'], dtype=np.float32)
        return encoded_feature

class Test_Dataset_Plus(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patent = self.df.iloc[idx]
        P1 = next(iter(patent.Patent1.values()))
        P2 = next(iter(patent.Patent2.values()))
        label = patent.Label  # No need to access .values here
        return np.array(P1, dtype=np.float32), np.array(P2, dtype=np.float32), label