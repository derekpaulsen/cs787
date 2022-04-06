from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd


class Optimizer(ABC):
    
    @abstractproperty
    def results(self):
        pass

    @abstractmethod
    def optimze(self, constraints):
        pass
    
    @staticmethod
    def create_results(constraints, weights, method_name):
        violated = constraints.mul(weights).sum(axis=1).sub(1.0).gt(0.0)
        hist = violated.groupby(level=0).sum().value_counts()
        
        return {
                'method_name' : method_name,
                'total_violated' : violated.sum(),
                'hist' : hist.to_dict(),
                'max_violated' : np.max(hist.index.values)
        }

    @staticmethod
    def format_constraints(constraints):
        # begins as (id2, matching tuple, id1)
        constraints.index = pd.MultiIndex.from_tuples(list(map(eval, constraints.index)))
        constraints.columns = pd.MultiIndex.from_tuples(list(map(eval, constraints.columns)))
        return constraints
    
    @staticmethod
    def _truncate_topk(grp, k):
        if len(grp) <= k:
            return grp

        # take the topk based on the sum of the scores
        take = grp.sum(axis=1)\
                    .sort_values()\
                    .tail(k)\
                    .index

        return grp.loc[take]


    @staticmethod
    def truncate_topk(constraints, k):
        return constraints.grouby(level=0)\
                .apply(lambda x : Optimizer._truncate_topk(x, k))
    
