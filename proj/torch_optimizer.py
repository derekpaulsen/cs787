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
        self.weights = nn.Parameter(torch.rand(self._dim, dtype=torch.float32, requires_grad=True) + .1)

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

    def __init__(self, iters=500, step_interval=25, timeout=Optimizer.TIMEOUT):
        self._timeout = timeout
        self._iters = iters
        # number iterations per round (gradient updates)
        self._step_interval = step_interval
        self._results = {
                'iters' : iters, 
                'step_interval' : step_interval,
                'total_grad_updates' : iters,
                'timeout' : self._timeout
        }
    
    @property
    def results(self):
        return self._results

    def optimize(self, constraints):
        timer = Timer()
        X = torch.Tensor(constraints.values)
        label = -torch.ones(len(constraints), dtype=torch.float32)
        cands = []
        self._results['setup_time'] = timer.get_total()
        while timer.get_total() < self._timeout:
            w, obj_val = self._optimize(constraints.columns, X, label)
            cands.append((w, obj_val, timer.get_total()))
        
        self._results['opt_time'] = timer.get_interval()
        cands_df = pd.DataFrame(cands, columns=['weights', 'obj_val', 'time']).sort_values('time')

        best_row = cands_df.sort_values('obj_val')\
                        .iloc[0]
    
        boost_map = pd.Series(
                np.maximum(best_row['weights'], 0.0),
                index=constraints.columns
        )
        boost_map = boost_map[boost_map.gt(0)]
        boost_map /= boost_map.min()
        weights = self.post_process_boost_map(boost_map)

        time_series = cands_df[['time', 'obj_val']]
        # take the running best
        time_series['obj_val'] = np.minimum.accumulate(time_series['obj_val'].values)

        self._results.update(Optimizer.create_results(constraints, weights, 'torch', time_series))

        return weights


    def _optimize(self, columns, X, label):
        
        model = BoostModel(len(columns))
        loss_fn = nn.L1Loss()
        optimizer = torch.optim.Adam(model.parameters(), lr=.5, weight_decay=0.0)
        lr_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        

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
            #p = torch.sum(pred).cpu().detach().numpy()
            cv = torch.count_nonzero(pred >= 0).cpu().detach().numpy()
            #log.debug(f'epoch {i} : sum = {p}, total > 0 = {cv}')
            weights[i] = model.weights
            cvs[i] = cv
        
        best_idx = np.argmin(cvs)
        log.debug(f'best idx : {best_idx}, value : {cvs[best_idx]}, {cvs}')

        return weights[best_idx], cvs[best_idx]
