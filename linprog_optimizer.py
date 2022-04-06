from utils import  get_index_name, read_table, Timer, get_logger, init_spark, invoke_task
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
sys.path.append('.')
pd.set_option('display.width', 150)
import pulp

N_PROCS=28
log = get_logger(__name__)

class LinProgOptimizer:
    pass
    COST_TYPES = {
            'max', 
            'sum'
    }

    def __init__(self, index, problem_cost='sum', query_cost='max', topk=25):

        self._problem_cost = problem_cost
        if self._problem_cost not in LinProgOptimizer.COST_TYPES:
            raise ValueError(f'problem_cost must be one of the following {LinProgOptimizer.COST_TYPES}')

        self._query_cost = query_cost
        if self._query_cost not in LinProgOptimizer.COST_TYPES:
            raise ValueError(f'query_cost must be one of the following {LinProgOptimizer.COST_TYPES}')

        # the topk query results to take for model creation
        self._topk = topk
        # solver config
        self._solver_name = 'PULP_CBC_CMD'
        self._solver_threads = multiprocessing.cpu_count() # use all threads on the solver
        self._solver_time_limit_secs = 600 # 10 mins
        self._solver_presolve = True 

        self.index = deepcopy(index)
        self.index.to_spark()
        init_spark()


    def _gen_fvs_row(self, search_doc, res_ids, is_match, query_spec):

        self.index.init()
        clauses = self.index.query_gen.generate_query_clauses(search_doc, query_spec)
        fvs = self.index.score_docs(res_ids.tolist(), clauses)
        fvs.rename(columns={'_id' : 'id1'}, inplace=True)

        fvs.set_index('id1', inplace=True)
        fvs.fillna(0.0, inplace=True)
        fvs['is_match'] = pd.Series(data=is_match, index=res_ids)
        fvs['id2'] = search_doc['_id']

        fvs = fvs.reset_index().set_index(['id2', 'is_match', 'id1'])
        fvs[('weight', '')] = 1.0
        fvs.columns = pd.MultiIndex.from_tuples(fvs.columns)
        
        fvs.sort_index(inplace=True)
        fvs.sort_index(axis=1, inplace=True)

        return fvs

    def _gen_fvs(self, search_table, res):

        recs = search_table.set_index(search_table['_id']).to_dict('index')
        qspec = self.index.get_full_query_spec(cross_fields=True)
        # TODO change this to query a spark dataframe directly
        # TODO execute entire pipeline here, all in one pandasinmaps function
        tasks = res.apply(lambda x : delayed(self._gen_fvs_row)(recs[x['id2']], x['id1_list'], x['is_match'], qspec), axis=1)

        fvs_chunks  = SparkContext.getOrCreate()\
                        .parallelize(tasks, len(tasks))\
                        .map(invoke_task)
        return fvs_chunks
    
    def _combine_constraints(self, df, keep_smallest, max_weight):
        # given a set a vectors x1, x2, x3, ... 
        # if keep_smallest
            # drop any vectors xi, s.t. xi'y >= xj'y, forall y != 0 and y >= 0
        # else 
            # drop any vectors xi, s.t. xi'y <= xj'y, forall y != 0 and y >= 0
        if len(df) < 2 or max_weight <= 1:
            return df
        
        comp = operator.le if keep_smallest else operator.ge
        df.sort_values(df.columns.values.tolist(), ascending=keep_smallest, inplace=True)
        i = 0
        #import pdb; pdb.set_trace()
        while i < len(df) - 1:
            idx = df.index[i]
            # rows which are less than or greater than 
            # than the current row being examined
            mask = comp(df.iloc[i], df.iloc[i+1:]).all(axis=1)
            if mask.any():
                cand_drop_rows = mask.index[mask.values].values
                # the weight of the new constraint if we combine the 
                # the first k constraints which are implied by the current
                weights = df[('weight' , '')]
                # take rows until the weight of the new constraint would be greater than max_weight
                take_to = weights.loc[cand_drop_rows].cumsum()\
                                .add(weights.at[idx])\
                                .le(max_weight)\
                                .sum()
                # combine the constraints
                if take_to > 0:
                    drop_rows = cand_drop_rows[:take_to]
                    df.at[idx, ('weight', '')] += weights[drop_rows].sum()
                    df.drop(index=drop_rows, inplace=True)
            i += 1

        return df

    def  _opt_lp(self, fvs_chunks, problem_cost_type):
        constraints = pd.concat(fvs_chunks.collect()).fillna(0.0)

        constraint_weights = constraints.pop('weight')

        # the variables corresponding to the boost values in the 
        # query spec, this is the the main optimization target
        boost_vars = pd.Series(data=[
            pulp.LpVariable(f'{x[0]}->{x[1]}', lowBound=0.0) for x in constraints.columns 
            ],
            index=constraints.columns
        )


        constraint_slack_vars = pd.Series(data=[
            pulp.LpVariable(f'y{i}_{j}_{k}', lowBound=0) for i,j,k in constraints.index
            ],
            index=constraints.index
        )
        # the costs for violating a given constraint
        constraint_costs = constraint_weights * constraint_slack_vars
        
        # the cost for a given matching tuple in a query
        #sub_query_index = pd.MultiIndex.from_frame(constraints.index.to_frame(index=None).iloc[:, 0:2].drop_duplicates())
        #sub_query_cost_vars = pd.Series(
        #        [pulp.LpVariable(f'y_{i}_{j}', lowBound=0) for i,j in sub_query_index],
        #        index = sub_query_index
        #).sort_index()
        # the total cost for a query
        query_cost_vars = pd.Series(
                [pulp.LpVariable(f'y_{i}', lowBound=0) for i in constraints.index.get_level_values(0).unique()],
                index = constraints.index.get_level_values(0).unique()
        )

        query_threshold_vars = pd.Series(
                [pulp.LpVariable(f't_{i}', lowBound=1.0) for i in constraints.index.get_level_values(0).unique()],
                index = constraints.index.get_level_values(0).unique()
        )
        

        problem = pulp.LpProblem('boosting_weights', pulp.LpMinimize)


        # add the constraints with indicator vars
        for idx, const in constraints.iterrows():
            # if the matching record score is >= the non-matching record(s)
            # then z_i = 0, else z_i > 0, increasing the cost of the solution
            # a_i'x - z_i <= -1
            # remove coeffecients = 0 
            is_match =idx[1]
            const = const[const != 0]
            v = const.mul(boost_vars[const.index]).sum() 
            query_threshold = query_threshold_vars.at[idx[0]]
            if is_match:
                problem += v >= query_threshold
            else:
                problem += v - constraint_slack_vars.at[idx] <= query_threshold - 1.0

        # add query costs
        for i, query_cost in query_cost_vars.items():
            query_cost = query_cost_vars[i]
            for const_cost in constraint_costs.loc[i]:
                problem += const_cost <= query_cost

        # add objective function
        if problem_cost_type == 'max':
            problem_cost = pulp.LpVariable('max_cost', lowBound=0)
            problem += problem_cost
            # add constraints to take the maximum query cost
            for query_cost in query_cost_vars:
                problem += query_cost <= problem_cost

        elif problem_cost_type == 'sum':
            problem += query_cost_vars.sum()

        else:
            raise ValueError()


        solver = self.get_solver()

        status = problem.solve(solver)
        
        log.info(pulp.LpStatus[status])
        # get values to boosting variables from solution
        boost_map = boost_vars.apply(pulp.value)

        filt = self._infer_filter(constraints, boost_map)

        return boost_map, filt

    def _infer_filter(self, constraints, boost_map):
        # remove the id of the query record, now just (is_match, indexed_record_id)
        constraints.index = constraints.index.droplevel()

        non_zeros = boost_map.index[boost_map.gt(0).values]
        # get only the columns used in the query plan
        # we only care about matches
        constraints = constraints[non_zeros].loc[True]
        boost_map = boost_map[non_zeros]
        
        cases = constraints.gt(0).drop_duplicates()
        log.info(f'cases for infering filter\n{cases}')
        covers = cases.all()
        covering_clauses = covers.index[covers.values].tolist()
        # if there are multiple covers, take the one that contributes the most to the 
        # scores of matches
        if len(covering_clauses) > 1:
            # get the weights for the various fields
            per_column_sums = constraints.mul(boost_map).sum()
            percent_contributions = per_column_sums / per_column_sums.sum()
            log.info(f'percent contributions to scores of matching tuples :\n{percent_contributions}')
        
            covering_clauses = [percent_contributions[covering_clauses].idxmax()]


        # get the (search -> index field) pairs where all matching tuples
        # have score > 0 (meaning they overlap on at least one token)
        log.info(f'filters inferred : {covering_clauses}')

        return covering_clauses

        

        
    def  _opt_mip_2(self, fvs_chunks, problem_cost_type):
        constraints = pd.concat(fvs_chunks.collect()).fillna(0.0)

        constraint_weights = constraints.pop('weight')

        # the variables corresponding to the boost values in the 
        # query spec, this is the the main optimization target
        boost_vars = pd.Series(data=[
            pulp.LpVariable(f'{x[0]}->{x[1]}', lowBound=0.0) for x in constraints.columns 
            ],
            index=constraints.columns
        )


        # binary vars used to indicate that the constraint has been violated
        constraint_indicator_vars = pd.Series(data=[
            pulp.LpVariable(f'y{i}_{j}_{k}', cat=pulp.const.LpBinary) for i,j,k in constraints.index
            ],
            index=constraints.index
        )
        # the costs for violating a given constraint
        constraint_costs = constraint_weights * constraint_indicator_vars
        
        # the cost for a given matching tuple in a query
        #sub_query_index = pd.MultiIndex.from_frame(constraints.index.to_frame(index=None).iloc[:, 0:2].drop_duplicates())
        #sub_query_cost_vars = pd.Series(
        #        [pulp.LpVariable(f'y_{i}_{j}', lowBound=0) for i,j in sub_query_index],
        #        index = sub_query_index
        #).sort_index()
        # the total cost for a query
        query_cost_vars = pd.Series(
                [pulp.LpVariable(f'y_{i}', lowBound=0) for i in constraints.index.get_level_values(0).unique()],
                index = constraints.index.get_level_values(0).unique()
        )

        query_threshold_vars = pd.Series(
                [pulp.LpVariable(f't_{i}', lowBound=1) for i in constraints.index.get_level_values(0).unique()],
                index = constraints.index.get_level_values(0).unique()
        )
        
        problem = pulp.LpProblem('boosting_weights', pulp.LpMinimize)


        # add the constraints with indicator vars
        for idx, const in constraints.iterrows():
            # if the matching record score is >= the non-matching record(s)
            # then z_i = 0, else z_i > 0, increasing the cost of the solution
            # a_i'x - z_i <= -1
            # remove coeffecients = 0 
            is_match =idx[1]
            const = const[const != 0]
            v = const.mul(boost_vars[const.index]).sum() 
            query_threshold = query_threshold_vars.at[idx[0]]
            if is_match:
                problem += v >= query_threshold
            else:
                problem += v - 1000 * constraint_indicator_vars.at[idx] <= query_threshold - 1.0

        # add query costs
        for i, query_cost in query_cost_vars.items():
            query_cost = query_cost_vars[i]
            problem += constraint_costs.loc[i].sum() == query_cost

        # add objective function
        if problem_cost_type == 'max':
            problem_cost = pulp.LpVariable('max_cost', lowBound=0)
            problem += problem_cost
            # add constraints to take the maximum query cost
            for query_cost in query_cost_vars:
                problem += query_cost <= problem_cost

        elif problem_cost_type == 'sum':
            problem += query_cost_vars.sum()

        else:
            raise ValueError()


        solver = self.get_solver()

        status = problem.solve(solver)
        
        log.info(pulp.LpStatus[status])
        # get values to boosting variables from solution
        boost_map = boost_vars.apply(pulp.value)

        filt = self._infer_filter(constraints, boost_map)

        return boost_map, filt

    def get_solver(self):
        return pulp.get_solver(
                self._solver_name,
                threads=self._solver_threads,
                timeLimit=self._solver_time_limit_secs,
                presolve=self._solver_presolve
        )

    def _truncate_training_query_res(self, row, k):
        if len(row['id1_list']) > k:
            for c in ['id1_list', 'is_match']:
                row[c] = row[c][:k]
        return row


    def optimize(self, search_table, res):
        # remove rows that cannot be used for the optimizer
        res = res.loc[res['id1_list'].apply(len) > 0]
        res = res.loc[res['is_match'].apply(np.any)]
        # truncate for very large input
        if len(res) > 100000:
            res = res.head(100000)
        # truncate the results to reduce the number of constraints in the 
        # optimization problem
        res = res.apply(lambda x : self._truncate_training_query_res(x, self._topk), axis=1)
        # a pyspark rdd of dataframes with the feature vectors
        fvs_chunks = self._gen_fvs(search_table, res)

        boost_map, filt = self._opt_lp(fvs_chunks, self._problem_cost)
        #boost_map = self. _opt_mip_2(fvs_chunks, self._problem_cost)
        log.info(f'boost map from lp before processing :\n{boost_map}')
        # remove all field -> indexed field pairs with boost > 0
        boost_map = boost_map.loc[boost_map.gt(0.0)]
        # normalize the boosts such that the min is 1
        boost_map /= boost_map.min()
        log.info(f'boost map :\n{boost_map}')
        # create query spec with non-zero boosting weights 
        bindex = boost_map.index.to_frame()
        qspec = QuerySpec(
                bindex.groupby(bindex.iloc[:,0])\
                        .apply(lambda x : set(x.iloc[:,1]))\
                        .to_dict()
        )
        # set boosting weights for the clauses
        qspec.boost_map = boost_map.to_dict()
        # add the filter to the queries for performance
        qspec.filter = filt

        return qspec





# old optimizer code 
# the number of constraints was greatly increased by the number of matches
# meaning that this rarely terminated with an optimal solution in the 
# time alloted to the optimizer
#    def  _opt_mip(self, fvs_chunks, problem_cost_type, query_cost_type):
#        constraints = self._make_constraints(fvs_chunks)
#
#        constraint_weights = constraints.pop('weight')
#
#        # the variables corresponding to the boost values in the 
#        # query spec, this is the the main optimization target
#        boost_vars = pd.Series(data=[
#            pulp.LpVariable(f'{x[0]}->{x[1]}', lowBound=0.0) for x in constraints.columns 
#            ],
#            index=constraints.columns
#        )
#
#
#        # binary vars used to indicate that the constraint has been violated
#        constraint_indicator_vars = pd.Series(data=[
#            pulp.LpVariable(f'y{i}_{j}_{k}', cat=pulp.const.LpBinary) for i,j,k in constraints.index
#            ],
#            index=constraints.index
#        )
#        # the costs for violating a given constraint
#        constraint_costs = constraint_weights * constraint_indicator_vars
#        
#        # the cost for a given matching tuple in a query
#        sub_query_index = pd.MultiIndex.from_frame(constraints.index.to_frame(index=None).iloc[:, 0:2].drop_duplicates())
#        sub_query_cost_vars = pd.Series(
#                [pulp.LpVariable(f'y_{i}_{j}', lowBound=0) for i,j in sub_query_index],
#                index = sub_query_index
#        ).sort_index()
#        # the total cost for a query
#        query_cost_vars = pd.Series(
#                [pulp.LpVariable(f'y_{i}', lowBound=0) for i in constraints.index.get_level_values(0).unique()],
#                index = constraints.index.get_level_values(0).unique()
#        )
#        
#        problem = pulp.LpProblem('boosting_weights', pulp.LpMinimize)
#
#        # add the constraints with indicator vars
#        for idx, const in constraints.iterrows():
#            # if the matching record score is >= the non-matching record(s)
#            # then z_i = 0, else z_i > 0, increasing the cost of the solution
#            # a_i'x - z_i <= -1
#            # remove coeffecients = 0 
#            const = const[const != 0]
#            problem += const.mul(boost_vars[const.index]).sum() - 1000 * constraint_indicator_vars.at[idx] <= -1.0
#        
#        # add query costs
#        for i, query_cost in query_cost_vars.items():
#            query_cost = query_cost_vars[i]
#            # add constriants to compute the sub query costs
#            for j, sub_query_cost in sub_query_cost_vars.loc[i].items():
#                problem += constraint_costs.loc[(i,j,)].sum() - sub_query_cost <= 0
#            
#            if query_cost_type == 'max':
#                for sub_query_cost in sub_query_cost_vars.loc[i]:
#                    problem += sub_query_cost - query_cost <= 0.0
#            elif query_cost_type == 'sum':
#                problem += sub_query_cost_vars.loc[i].sum() - query_cost <= 0.0
#            else:
#                raise ValueError()
#
#        # add objective function
#        if problem_cost_type == 'max':
#            problem_cost = pulp.LpVariable('max_cost', lowBound=0)
#            problem += problem_cost
#            # add constraints to take the maximum query cost
#            for query_cost in query_cost_vars:
#                problem += query_cost - problem_cost <= 0.0
#
#        elif problem_cost_type == 'sum':
#            problem += query_cost_vars.sum()
#
#        else:
#            raise ValueError()
#
#
#        solver = self.get_solver()
#
#        status = problem.solve(solver)
#        
#        log.info(pulp.LpStatus[status])
#        # get values to boosting variables from solution
#        boost_map = boost_vars.apply(pulp.value)
#
#        return boost_map
#

    #def _make_constraints_group(self, fvs):
    #    id2 = fvs.index.levels[0].unique()[0]
    #    fvs.reset_index(level=[0], drop=True, inplace=True)
    #    # need both matches and non-matches to make constraints
    #    if True not in fvs.index or False not in fvs.index:
    #        return pd.DataFrame(columns=fvs.columns)
    #    fvs.loc[(True, ), ('weight', )] = 0.0

    #    matches = fvs.loc[True]
    #    non_matches = fvs.loc[False]
    #    constraints_parts = []

    #    for idx, row in matches.iterrows():
    #        p = non_matches - row
    #        # drop non-matches that cannot be satisfied
    #        p = p.loc[p.lt(0).any(axis=1)]
    #        p = self._combine_constraints(p, False, self._max_constraint_weight)
    #        index = p.index.to_frame()
    #        index.columns = ['non_match_id']
    #        index.insert(0, 'match_id', idx)
    #        index.insert(0, 'search_id', id2)
    #        p.index = pd.MultiIndex.from_frame(index)

    #        constraints_parts.append(p)

    #    constraints = pd.concat(constraints_parts)
    #    # drop any constraint where all coeffecients are negative
    #    # which means that all the matching tuple is strictly bigger than the 
    #    # non-matching tuple, excludeing the weight column which is always positive
    #    constraints = constraints.loc[constraints.gt(0).sum(axis=1) > 1]
    #    #index = (id of query record, id of matching record, id of non-matching record)

    #    return constraints


    #def _make_constraints(self, fvs_chunks, max_constraint_groups=5000):
    #    constraints = fvs_chunks.map(self._make_constraints_group)\
    #                            .filter(lambda x : len(x) > 0)\
    #                            .take(max_constraint_groups)

    #    constraints = pd.concat(constraints).fillna(0.0)
    #    

    #    return constraints
    #
