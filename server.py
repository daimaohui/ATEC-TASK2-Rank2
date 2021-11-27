from datetime import datetime
import os
import shutil
import unittest
import pickle
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F
from context import FederatedSGD
from context import PytorchModel
from learning_model import Linear
from preprocess import get_test_data
import pandas as pd
from sklearn import preprocessing
class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir, resultdir):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.worker_info = {}
        self.current_round_grads = []
        self.init_model_path = init_model_path
        self.aggr = FederatedSGD(
            model=PytorchModel(torch=torch,
                               model_class=Linear,
                               init_model_path=self.init_model_path,
                               optim_name='AdamW'),
            framework='pytorch',
        )
        self.testworkdir = testworkdir
        self.RESULT_DIR = resultdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)
        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.test_data, self.test_edges = get_test_data()

        self.preprocess_test_data()

        self.round_train_acc = []
        self.model=None
        self.optimizer=None
        self.model_1=None
        self.optimizer_1=None
    def preprocess_test_data(self):
        return
        # self.predict_data = self.test_data[self.test_data['class'] == 3]  # to be predicted
        # self.predict_data_txId = self.predict_data[['txId', 'Timestep']]
        # x = self.predict_data.iloc[:, 3:]
        # # x = x.fillna(method='ffill')
        # x = x.reset_index(drop=True)
        # x = x.to_numpy().astype(np.float32)
        # x[x == np.inf] = 1.
        # x[np.isnan(x)] = 0.
        # features = sp.csr_matrix(x, dtype=np.float32)
        # # build graph
        # idx = np.array(self.predict_data['txId'])
        # idx_map = {j: i for i, j in enumerate(idx)}
        # edges_unordered = np.array(self.test_edges, dtype=np.int32)
        # t = []
        # a = [i for i in range(len(idx))]
        # h = len(idx)
        # for i in edges_unordered:
        #     if idx_map.get(i[0]) != None and idx_map.get(i[1]):
        #         t.append([idx_map.get(i[0]), idx_map.get(i[1])])
        #         x, a = find_father(idx_map.get(i[0]), a)
        #         y, a = find_father(idx_map.get(i[1]), a)
        #         if x != y:
        #             h -= 1
        #             a[y] = x
        # print("____________" + str(h) + "____________________")
        # edges = np.array(t)
        # # edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)
        # # print(edges)
        # adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        #                     shape=(features.shape[0], features.shape[0]),
        #                     dtype=np.float32)
        # #
        # # # build symmetric adjacency matrix
        # adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # features = normalize_features(features)
        # adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        # return features, adj

    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads):  # receive grads info from worker
        self.current_round_grads.append(grads)

    def receive_worker_info(self, info):  # receive worker info from worker
        self.worker_info = info

    def process_round_train_acc(self):  # process the "round_train_acc" info from worker
        self.round_train_acc.append(self.worker_info["train_acc"])

    def print_round_train_acc(self):
        mean_round_train_acc = np.mean(self.round_train_acc) * 100
        print("\nMean_round_train_acc: ", "%.2f%%" % (mean_round_train_acc))
        self.round_train_acc = []
        return {"mean_round_train_acc": mean_round_train_acc
                }

    def aggregate(self):
        self.aggr(self.current_round_grads)

        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1
        self.current_round_grads = []

        return info

    def save_prediction(self, predition):
        predition.to_csv(os.path.join(self.RESULT_DIR, 'result.csv'), index=0)

    def save_model(self, model,optimizer):
        torch.save({'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()},'/tmp/competetion-test/model.pth')

    def save_testdata_prediction(self, model,model_1, device, test_batch_size):
        # print(model.ty)
        print(self.test_edges)
        self.predict_data_txId = pd.DataFrame()
        num = self.test_data.loc[:, 'Timestep'].max()
        Times = []
        txId = []
        predict = []
        for ti in range(0,num+1):
            predict_data_1=self.test_data[self.test_data['Timestep']==ti]
            x = predict_data_1.iloc[:, 3:]
            x = x.reset_index(drop=True)
            x = x.to_numpy().astype(np.float32)
            x[x == np.inf] = 1.
            x[np.isnan(x)] = 0.
            features = x
            features = torch.FloatTensor(np.array(features))
            pred = model(features)
            t = []
            for i in pred:
                t.append(int(torch.argmax(i)))
            idx = np.array(predict_data_1['txId'])
            idx_map = {j: i for i, j in enumerate(idx)}
            edges = np.array(self.test_edges)
            for j in range(2):
                h = [0. for i in range(len(idx))]
                h_1 = [0. for i in range(len(idx))]
                for i in edges:
                    if i[2]==ti:
                        if t[idx_map.get(i[0])]==0:
                            h[idx_map.get(i[1])]+=2
                        else:
                            h[idx_map.get(i[1])] += 1
                        h_1[idx_map.get(i[1])]+=1
                        if t[idx_map.get(i[1])]==0:
                            h[idx_map.get(i[0])]+=2
                        else:
                            h[idx_map.get(i[0])] += 1
                        h_1[idx_map.get(i[0])]+=1
                for i in range(len(h)):
                    if h_1[i] == 0:
                        h[i] = 0
                    else:
                        h[i] = h[i] / h_1[i]
                predict_data_1['edges']=h
                x = predict_data_1.iloc[:, 3:]
                x = x.reset_index(drop=True)
                x = x.to_numpy().astype(np.float32)
                x[x == np.inf] = 1.
                x[np.isnan(x)] = 0.
                features = x
                features = torch.FloatTensor(np.array(features))
                pred = model_1(features)
                t = []
                for i in pred:
                    t.append(int(torch.argmax(i)))
            predict_data_1['pre']=t
            predict_data_1 = predict_data_1[predict_data_1['class'] == 3]
            for j in predict_data_1['Timestep']:
                Times.append(j)
            for j in predict_data_1['txId']:
                txId.append(j)
            print(predict_data_1.shape)
            for i in predict_data_1['pre']:
                predict.append(i)
        self.predict_data_txId['txId'] = txId
        self.predict_data_txId['Timestep'] = Times
        self.predict_data_txId['prediction'] = predict
        self.save_prediction(self.predict_data_txId)

