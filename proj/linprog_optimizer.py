from utils import  get_logger, Timer
import multiprocessing
from optimizer import Optimizer
from tempfile import mkstemp

import pandas as pd
import sys
sys.path.append('.')
pd.set_option('display.width', 150)
import pulp

N_PROCS=28
log = get_logger(__name__)


def parse_log_line(l):
    toks = [x for x in l.split() if x[0].isdigit()]
    if len(toks) != 9:
        return None
    # e.g. 120s
    t = int(toks[-1][:-1])
    # typically a float
    soln_val = float(toks[5])

    return (t, soln_val)


def parse_gurobi_log(f):
    with open(f) as ifs:
        res = list(filter(lambda x : x is not None, map(parse_log_line, ifs)))
        df = pd.DataFrame(res, columns=['time', 'obj_val'])

    return df



class LinProgOptimizer(Optimizer):

    def __init__(self, relaxation):
        self._relaxed=relaxation
        # solver config
        self._solver_name = 'GUROBI_CMD'
        self._solver_threads = multiprocessing.cpu_count() # use all threads on the solver
        # no timelimit currently
        self._solver_time_limit_secs = 100 
        #self._solver_time_limit_secs = 3600 # 60 mins
        self._solver_presolve = True 
        self._log_file = mkstemp(prefix='gurobi.log.')[1]

        self._results = {
                'solver' : self._solver_name,
                'relaxed' : self._relaxed,
                'log_file' : self._log_file
        }

    
    @property
    def results(self):
        return self._results

    def get_solver(self):
        return pulp.get_solver(
                self._solver_name,
                threads=self._solver_threads,
                timeLimit=self._solver_time_limit_secs,
                logPath=self._log_file
        )
    
    def _solve_problem(self, problem):
        timer = Timer()
        problem.solve(self.get_solver())
        return timer.get_interval()

    def  _opt(self, constraints, relaxed):
        # the variables corresponding to the boost values in the 
        # query spec, this is the the main optimization target
        boost_vars = pd.Series(data=[
            pulp.LpVariable(f'{x[0]}->{x[1]}', lowBound=0.0) for x in constraints.columns 
            ],
            index=constraints.columns
        )
        
        # switch between strict binary and relaxed LP
        if relaxed:
            kwargs = {'lowBound' : 0.0, 'upBound' : 1.0}
        else:
            kwargs = {'cat' : pulp.const.LpBinary}

        # binary vars used to indicate that the constraint has been violated
        constraint_indicator_vars = pd.Series(data=[
            pulp.LpVariable(f'y{i}_{j}_{k}', **kwargs) for i,j,k in constraints.index
            ],
            index=constraints.index
        )
        # the costs for violating a given constraint
        constraint_costs = constraint_indicator_vars
        
        # the cost for a given matching tuple in a query
        sub_query_index = pd.MultiIndex.from_frame(constraints.index.to_frame(index=None).iloc[:, 0:2].drop_duplicates())
        sub_query_cost_vars = pd.Series(
                [pulp.LpVariable(f'y_{i}_{j}', lowBound=0) for i,j in sub_query_index],
                index = sub_query_index
        ).sort_index()
        # the total cost for a query
        query_cost_vars = pd.Series(
                [pulp.LpVariable(f'y_{i}', lowBound=0) for i in constraints.index.get_level_values(0).unique()],
                index = constraints.index.get_level_values(0).unique()
        )
        
        problem = pulp.LpProblem('boosting_weights', pulp.LpMinimize)

        # add the constraints with indicator vars
        for idx, const in constraints.iterrows():
            # if the matching record score is >= the non-matching record(s)
            # then z_i = 0, else z_i > 0, increasing the cost of the solution
            # a_i'x - z_i <= -1
            # remove coeffecients = 0 
            const = const[const != 0]
            problem += const.mul(boost_vars[const.index]).sum() - 1000 * constraint_indicator_vars.at[idx] <= -1.0
        
        # add query costs
        for i, query_cost in query_cost_vars.items():
            query_cost = query_cost_vars[i]
            # add constriants to compute the sub query costs
            for j, sub_query_cost in sub_query_cost_vars.loc[i].items():
                problem += constraint_costs.loc[(i,j,)].sum() - sub_query_cost <= 0
            
            problem += sub_query_cost_vars.loc[i].sum() - query_cost <= 0.0

        # add objective function
        problem += query_cost_vars.sum()
        
        self._results['opt_time'] = self._solve_problem(problem)
        # get values to boosting variables from solution
        boost_map = boost_vars.apply(pulp.value)
        
        return boost_map


    def optimize(self, constraints):

        boost_map = self._opt(constraints, self._relaxed)
        boost_map = self.post_process_boost_map(boost_map)
        
        time_series = parse_gurobi_log(self._log_file)

        method = 'LP' if self._relaxed else 'MILP'
        self._results.update(Optimizer.create_results(constraints, boost_map, method, time_series))

        return boost_map

        
