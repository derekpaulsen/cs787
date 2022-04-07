import numpy as np
import pandas as pd
import sys
import torch
from optimizer import Optimizer
from torch import nn
from utils import get_logger, Timer


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



class TorchOptimizer(Optimizer):

    def __init__(self, iters=500, step_interval=10, restarts=100):
        self._restarts = restarts
        self._iters = iters
        # number iterations per round (gradient updates)
        self._step_interval = step_interval
        self._results = {
                'iters' : iters, 
                'step_interval' : step_interval,
                'total_grad_updates' : iters,
                'restarts' : restarts
        }
    
    @property
    def results(self):
        return self._results

    def optimize(self, constraints):
        timer = Timer()
        cands = [self._optimize(constraints) for i in range(self._restarts)]

        self._results['opt_time'] = timer.get_interval()
        # sort by the number of violated constraints
        cands.sort(key=lambda x : x[1])
        weights = self.post_process_boost_map(cands[0][0])
        self._results.update(Optimizer.create_results(constraints, weights, 'torch'))

        return weights


    def _optimize(self, constraints):
        
        model = BoostModel(len(constraints.columns))
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=0.0)
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        
        X = torch.Tensor(constraints.values)
        label = -torch.ones(len(constraints), dtype=torch.float32)

        niter = self._iters  // self._step_interval

        weights = [None] * niter
        cvs = np.zeros(niter, dtype=np.int32)
        for i in range(niter):
            for j in range(self._step_interval):
                optimizer.zero_grad()

                pred = model(X)
                loss = loss_fn(pred, label)
                loss.backward()

                optimizer.step()

            lr_decay.step()
            p = torch.sum(pred).cpu().detach().numpy()
            cv = torch.count_nonzero(pred > -1.0).cpu().detach().numpy()
            log.debug(f'epoch {i} : sum = {p}, total > 0 = {cv}')
            weights[i] = model.weights
            cvs[i] = cv
        
        best_idx = np.argmin(cvs)

        boost_map = pd.Series(
                np.maximum(weights[best_idx], 0.0),
                index=constraints.columns
        )
        boost_map = boost_map[boost_map.gt(0)]
        boost_map /= boost_map.min()

        return boost_map, cvs[best_idx]
