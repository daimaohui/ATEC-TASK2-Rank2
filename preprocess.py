import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

TRAINDATA_DIR = './TrainData/TrainData/'
TESTDATA_PATH = './Test_X/Test_X.pkl'

class CompDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self._data = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)



def get_user_data(user_idx):
    train_data_path  = TRAINDATA_DIR+'WorkerData_{}.csv'.format(user_idx)
    t_data = pd.read_csv(train_data_path)
    # read edges data
    train_edges_path = TRAINDATA_DIR+'WorkerDataEdges_{}.csv'.format(user_idx)
    t_edges = pd.read_csv(train_edges_path)
    return t_data,t_edges


def get_test_data():
    with open(TESTDATA_PATH, 'rb') as fin:
        data = pickle.load(fin)
    return data["data"],data["edges"]