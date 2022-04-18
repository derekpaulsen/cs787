from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import pandas as pd


class Optimizer(ABC):

    TIMEOUT=2700 # 24hrs of cpu time
    
    @abstractproperty
    def results(self):
        pass

    @abstractmethod
    def optimize(self, constraints):
        pass
    
    @staticmethod
    def create_results(constraints, weights, method_name , time_series):
        violated = constraints.mul(weights).sum(axis=1).sub(1.0).gt(0.0)
        hist = violated.groupby(level=0).sum().value_counts()
        weights.index = [str(x) for x in weights.index]
        return {
                'method_name' : method_name,
                'total_violated' : int(violated.sum()),
                'hist' : hist.to_dict(),
                'max_violated' : int(np.max(hist.index.values)),
                'boost_weights' : weights.to_dict(),
                'time_series' : time_series.to_dict(orient='list')
        }

    @staticmethod
    def read_constraints(file):
        constraints = pd.read_parquet(file)
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
        c = constraints.groupby(level=0)\
                .apply(lambda x : Optimizer._truncate_topk(x, k))

        if len(c.index.names) == 4:
            c.index = c.index.droplevel(0)
        return c

    @staticmethod
    def post_process_boost_map(boost_map):
        boost_map = boost_map.where(boost_map > 0).dropna()
        return boost_map / boost_map.min()
