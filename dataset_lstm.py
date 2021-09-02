import os, os.path
import pandas as pd
import torch
import numpy as np


class TimeSeriesDataset(torch.utils.data.Dataset):
    NUM_FILES = 266
    NUM_SAMPLES = 70
    NUM_FREQ = 9
        
    def __init__(self, freq_dir, split):
    
        self.freq_dir = freq_dir
        self.split = split
        self.NUM_FILES = len(os.listdir(freq_dir + '/' + str(100) + '/' + split))
       

    def __len__(self):
        return self.NUM_FILES*self.NUM_FREQ

    def __getitem__(self, idx):
    
        freq_idx = int(idx/self.NUM_FILES)
        freq_path = os.path.join(self.freq_dir, str(100 + freq_idx*50), self.split, str((idx % self.NUM_FILES) + 1) + ".csv")
        time_series = pd.read_csv(freq_path, header=None).to_numpy()
        time_series = time_series[:,:self.NUM_SAMPLES]
        #print(time_series[:,:self.NUM_SAMPLES])

        label = freq_idx
        return torch.tensor(time_series, dtype=torch.float), label
