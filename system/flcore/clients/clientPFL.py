import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics

from flcore.clients.model_conpression import sort_and_compress_model
from utils.data_utils import read_client_data
from utils.Local_aggregation import LocalAggregation
from multiprocessing import cpu_count
import openpyxl as op
import math
import os

class clientPFL(object):
    def __init__(self, args, id, train_samples, test_samples):
        self.upload_payload = None
        self.model = copy.deepcopy(args.model)
        self.model_before = None
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.args = args

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_steps = args.local_steps

        self.loss = nn.CrossEntropyLoss()
        self.layer_idx = args.layer_idx
        self.alpha =[]
        self.J=[]

        self.wb = op.Workbook()
        self.ws = self.wb['Sheet']

        if args.model_str == "cnn":
            # CNN
            params = [
                {'params': list(self.model.parameters())[:2]},
                {'params': list(self.model.parameters())[2:4]},
                {'params': list(self.model.parameters())[4:6]},
                {'params': list(self.model.parameters())[6:8]},
            ]

        elif args.model_str == "resnet":
            # Resnet
            params = [
                {'params': self.model.conv1.parameters()},
                {'params': self.model.bn1.parameters()},
                {'params': self.model.layer1.parameters()},
                {'params': self.model.layer2.parameters()},
                {'params': self.model.layer3.parameters()},
                {'params': self.model.layer4.parameters()},
                {'params': self.model.fc.parameters()}
            ]

        elif args.model_str == "fastText":
            # fastText
            params = [
                {'params': list(self.model.parameters())[:1]},
                {'params': list(self.model.parameters())[1:3]},
                {'params': list(self.model.parameters())[3:5]},
            ]

        self.optimizer = torch.optim.SGD(params)
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.local_aggregation = LocalAggregation(self.layer_idx)

    def local_initialization(self, received_global_model, acc):


        self.local_aggregation.adaptive_local_aggregation(received_global_model, self.model, acc)



    def train(self,alpha,J):
        self.model_before = copy.deepcopy(self.model)

        #alpha,J=decide()
        alpha=alpha
        J=J
        model_small, info = sort_and_compress_model(self.model_before, alpha=alpha, layer_idx=self.layer_idx)

        # 1) 随机采样 J 个样本
        full_train_data = read_client_data(self.dataset, self.id, is_train=True)
        J = min(J, len(full_train_data))
        # J=len(full_train_data)
        import random
        indices = list(range(len(full_train_data)))
        random.shuffle(indices)
        selected_indices = indices[:J]
        sampled_data = [full_train_data[i] for i in selected_indices]
        # 使用固定物理 batch size 跑 SGD
        trainloader = DataLoader(
            sampled_data,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
        )
        model_small.train()
        sub_optimizer = torch.optim.SGD(model_small.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        # 3) 在子模型上做本地训练
        for _ in range(self.local_steps):
            for x, y in trainloader:
                if isinstance(x, list):
                    x = [t.to(self.device) for t in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                sub_optimizer.zero_grad()
                output = model_small(x)
                loss = self.loss(output, y)
                loss.backward()
                sub_optimizer.step()

        self.upload_payload = {
            "id": self.id,
            "state_dict": copy.deepcopy(model_small.state_dict()),
            "alpha": float(alpha),
            "J": int(J),
            "info": info
        }
        return self.upload_payload






    # def decide(self):

        # return alpha,J

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=False)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=True, shuffle=False)

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self, model=None):
        trainloader = self.load_train_data()
        if model == None:
            model = self.model
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


