import pulp
from utils import get_index_name, read_table, Timer, get_logger, init_spark, invoke_task
from pyspark import SparkContext
import operator
import multiprocessing
import pyspark.sql.functions as F
from joblib import delayed


import numpy as np
from tqdm import tqdm
from es.query_generator import QuerySpec
import pandas as pd
from copy import deepcopy
import sys
import torch
from torch import nn
sys.path.append('.')
pd.set_option('display.width', 150)

N_PROCS = 28
log = get_logger(__name__)



class PosLinear(nn.Module):

    def __init__(self, in_features):
        nn.Module.__init__(self)

        self._dim = in_features
        self.weights = nn.Parameter(torch.rand(self._dim, dtype=torch.float32, requires_grad=True))

    def forward(self, X):
        return torch.matmul(X, nn.functional.relu(self.weights))


class BoostModel(nn.Module):
    def __init__(self, in_features):
        nn.Module.__init__(self)
        
        self._dim = in_features
        self.seq = nn.Sequential( 
                    PosLinear(in_features),
                    nn.Hardtanh()
        )

    def forward(self, X):
        return self.seq(X)

    @property
    def weights(self):
        return self.seq[0].weights.cpu().detach().numpy()



class TorchOptimizer:

    def __init__(self, rounds=50, round_duration=10, topk=50, restarts=10):
        self._topk = topk
        self._restarts = restarts
        self._rounds = rounds
        # number iterations per round (gradient updates)
        self._round_duration = round_duration
        self.results_ = {
                'topk' : topk, 
                'rounds' : rounds, 
                'round_duration' : round_duration,
                'total_grad_updates' : rounds * round_duration,
                'restarts' : restarts
        }
    
    

    def optimize(self, constraints):
        pass


    def _optimize(self, constraints):
        
        model = BoostModel(len(constraints.columns))
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.0)
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        
        X = torch.Tensor(constraints.values)
        label = -torch.ones(len(constraints), dtype=torch.float32)
        
        for i in range(self._rounds):
            for j in range(self._round_duration):
                optimizer.zero_grad()

                pred = model(X)
                loss = loss_fn(pred, label)
                loss.backward()

                optimizer.step()
            
            lr_decay.step()
            p = torch.sum(pred).cpu().detach().numpy()
            cv = torch.count_nonzero(pred > 0).cpu().detach().numpy()
            log.info(f'epoch {i} : sum = {p}, total > 0 = {cv}')

        boost_map = pd.Series(
                model.weights,
                index=constraints.columns
        )
        return boost_map
