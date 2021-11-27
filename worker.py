import torch
import torch.nn.functional as F
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data
import scipy.sparse as sp
from preprocess import CompDataset
from preprocess import get_user_data
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import f1_score

from torch.autograd import Variable
from sklearn import preprocessing

class FocalLoss(torch.nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            alpha = np.array(alpha)
            alpha = torch.from_numpy(alpha)
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        # P = F.softmax(inputs)
        P = inputs
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, device):
        self.x = torch.from_numpy(x).to(torch.float32)
        self.y = torch.from_numpy(y)
        self.device = device

    def __getitem__(self, index):
        xi = self.x[index]
        yi = self.y[index]
        return xi, yi

    def __len__(self):
        return len(self.y)


class Worker(object):
    def __init__(self, user_idx):
        self.user_idx = user_idx
        self.data, self.edges = get_user_data(self.user_idx)  # The worker can only access its own data
        # print(self.data)
        print(self.data.shape)
        print(self.edges.shape)
        # print(self.edges)
        self.ps_info = {}
        self.label_temp=[]

    def preprocess_worker_data(self):  # dui
        self.train_id = len(self.data[self.data['class'] == 1])
        self.test_id = len(self.data[self.data['class'] == 0])
        self.data_1 = self.data[self.data['class'] != 2]
        x = self.data_1.iloc[:, 2:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.features = x
        y = self.data_1['class']
        self.labels = y.reset_index(drop=True)
        self.labels = np.array(self.labels, dtype=np.float32)

    def receive_server_info(self, info):  # receive info from PS
        self.ps_info = info

    def process_mean_round_train_acc(self):  # process the "mean_round_train_acc" info from server
        mean_round_train_acc = self.ps_info["mean_round_train_acc"]
        # You can go on to do more processing if needed

    def user_round_train(self, model, device, n_round, batch_size, optimizer,n_round_samples=-1, debug=False):
        dataset = MyDataset(self.features, self.labels, torch.device("cpu:0"))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        criterion = FocalLoss(class_num=2,alpha=[0.8,0.2],gamma=2)
        # 训练次数设为10次
        for epoch in range(1):
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                # 训练模型
                optimizer.zero_grad()
                outputs = model(inputs)
                labels = labels.long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # 输出模型当前状态
                #print("loss:"+str(loss.item()))
        criterion=None
        label=[]
        t=[]
        features = torch.from_numpy(self.features).to(torch.float32)
        for i in self.labels:
            label.append(i)
        pred = model(features)
        for i in pred:
            t.append(int(torch.argmax(i)))
        self.label_temp=t
        TP = 0.
        TN = 0.
        FP = 0.
        FN = 0.
        for i, j in zip(label, t):
            if i == 1 and j == 1:
                TP += 1
            if i == 0 and j == 0:
                TN += 1
            if i == 0 and j == 1:
                FP += 1
            if i == 1 and j == 0:
                FN += 1
        if TN != 0:
            precision = TN / (TN + FN)
            recall = TN / (TN + FP)
        else:
            precision = 0.
            recall = 0.
        if TN != 0:
            score = 2 * precision * recall / (precision + recall)
        else:
            score = 0
        f1 = score
        grads = {'n_samples': self.features.shape[0], 'named_grads': {}}
        for name, param in model.named_parameters():
            grads['named_grads'][name] = param.grad.detach().cpu().numpy()

        worker_info = {}
        worker_info["train_acc"] = f1
        return grads, worker_info, model, f1,optimizer
    def createbyedges(self,model):
        x = self.data.iloc[:, 2:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.features_1 = x
        features_1 = torch.from_numpy(self.features_1).to(torch.float32)
        pred=model(features_1)
        t=[]
        for i in pred:
            t.append(int(torch.argmax(i)))
        h=[0. for i in range(len(self.data))]
        h_1 = [0. for i in range(len(self.data))]
        edges_unordered = np.array(self.edges, dtype=np.int32)
        for i in edges_unordered:
            if t[i[0]]==0:
                h[i[1]]+=2
            else:
                h[i[1]]+=1
            h_1[i[1]]+=1
            if t[i[1]]==0:
                h[i[0]]+=2
            else:
                h[i[0]]+=1
            h_1[i[0]]+=1
        for i in range(len(h)):
            if h_1[i]==0:
                h[i]=0
            else:
                h[i]=h[i]/h_1[i]
        self.data['edges']=h
        self.data_1 = self.data[self.data['class'] != 2]
        x = self.data_1.iloc[:, 2:]
        x = x.reset_index(drop=True)
        x = x.to_numpy().astype(np.float32)
        x[x == np.inf] = 1.
        x[np.isnan(x)] = 0.
        self.features = x
        y = self.data_1['class']
        self.labels = y.reset_index(drop=True)
        self.labels = np.array(self.labels, dtype=np.float32)
    def trainbyedges(self,model_1,optimizer_1):
        dataset = MyDataset(self.features, self.labels, torch.device("cpu:0"))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
        criterion = FocalLoss(class_num=2, alpha=[0.8, 0.2], gamma=2)
        # 训练次数设为10次
        for epoch in range(1):
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data
                # 训练模型
                optimizer_1.zero_grad()
                outputs = model_1(inputs)
                labels = labels.long()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_1.step()
                # 输出模型当前状态
                # print("loss:"+str(loss.item()))
        criterion = None
        label = []
        t = []
        features = torch.from_numpy(self.features).to(torch.float32)
        for i in self.labels:
            label.append(i)
        pred = model_1(features)
        for i in pred:
            t.append(int(torch.argmax(i)))
        self.label_temp = t
        TP = 0.
        TN = 0.
        FP = 0.
        FN = 0.
        for i, j in zip(label, t):
            if i == 1 and j == 1:
                TP += 1
            if i == 0 and j == 0:
                TN += 1
            if i == 0 and j == 1:
                FP += 1
            if i == 1 and j == 0:
                FN += 1
        if TN != 0:
            precision = TN / (TN + FN)
            recall = TN / (TN + FP)
        else:
            precision = 0.
            recall = 0.
        if TN != 0:
            score = 2 * precision * recall / (precision + recall)
        else:
            score = 0
        f1 = score
        return f1,model_1,optimizer_1
    def train(self, model_1, optimizer_1):
        features_1 = np.insert(self.features, 165, values=np.array(self.label_temp), axis=1)
        dataset_1 = MyDataset(features_1, self.labels, torch.device("cpu:0"))
        data_loader_1 = torch.utils.data.DataLoader(dataset_1, batch_size=128, shuffle=True)
        criterion_1 = FocalLoss(class_num=2, alpha=[0.8, 0.2], gamma=2)
        for epoch in range(1):
            for i, data_1 in enumerate(data_loader_1, 0):
                inputs_1, labels_1 = data_1
                # 训练模型
                optimizer_1.zero_grad()
                outputs_1 = model_1(inputs_1)
                labels_1 = labels_1.long()
                loss_1 = criterion_1(outputs_1, labels_1)
                loss_1.backward()
                optimizer_1.step()
                # 输出模型当前状态
        criterion_1=None
        t_1 = []
        label_1=[]
        for i in self.labels:
            label_1.append(i)
        features_1 = torch.from_numpy(features_1).to(torch.float32)
        pred = model_1(features_1)
        for i in pred:
            t_1.append(int(torch.argmax(i)))
        TP = 0.
        TN = 0.
        FP = 0.
        FN = 0.
        for i, j in zip(label_1, t_1):
            if i == 1 and j == 1:
                TP += 1
            if i == 0 and j == 0:
                TN += 1
            if i == 0 and j == 1:
                FP += 1
            if i == 1 and j == 0:
                FN += 1
        if TN != 0:
            precision = TN / (TN + FN)
            recall = TN / (TN + FP)
        else:
            precision = 0.
            recall = 0.
        # print(precision)
        # print(recall)
        if TN != 0:
            score = 2 * precision * recall / (precision + recall)
        else:
            score = 0
        # print(score)
        f1 = score
        return  f1,model_1, optimizer_1