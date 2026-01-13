import copy
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from typing import Dict, List, Optional, Tuple
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
        self.learning_rate = args.local_learning_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                                         weight_decay=1e-4)
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


        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.local_aggregation = LocalAggregation(self.layer_idx)

    def local_initialization(self, received_global_model, acc):
        # for local_m, global_m in zip(self.model.modules(), received_global_model.modules()):
        #     if isinstance(local_m, nn.Conv2d):
        #         local_m.in_channels = global_m.in_channels
        #         local_m.out_channels = global_m.out_channels
        #     elif isinstance(local_m, nn.Linear):
        #         local_m.in_features = global_m.in_features
        #         local_m.out_features = global_m.out_features
        #self.local_aggregation.adaptive_local_aggregation(received_global_model, self.model, acc)
        self.model=received_global_model



    def train(self,alpha,J):


        #alpha,J=decide()
        alpha=1.0
        J=J


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
            full_train_data,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=True,
        )
        # sub_model = sort_and_compress_model(
        #     self.model,
        #     alpha=float(alpha),
        #     skip_last=int(self.layer_idx),
        # ).to(self.device)

        self.model.train()
        sub_optimizer = torch.optim.SGD( self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-4)
        # 3) 在子模型上做本地训练
        for _ in range(self.local_steps):
            for x, y in trainloader:
                if isinstance(x, list):
                    x = [t.to(self.device) for t in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                sub_optimizer.zero_grad()
                output =  self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                sub_optimizer.step()

        # self.model=sub_model
        self.upload_payload = {
            "id": self.id,
            "state_dict": copy.deepcopy( self.model.state_dict()),
            "alpha": float(alpha),
            "J": int(J)
        }
        return self.upload_payload

    def provide_upload_package(self) -> Optional[Dict]:
        """训练完成后由服务器调用，获取缓存的上传数据包。"""
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



