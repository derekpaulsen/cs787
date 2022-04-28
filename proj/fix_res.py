from argparse import ArgumentParser
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from torch_optimizer import TorchOptimizer
from linprog_optimizer import LinProgOptimizer
from optimizer import Optimizer
from pprint import pformat 
from utils import get_logger


log = get_logger(__name__)

DATA_DIR = Path('./data')
datasets = list(DATA_DIR.glob('*.parquet'))
argp = ArgumentParser()

argp.add_argument('--input', required=True)
argp.add_argument('--k', required=False, default=50, type=int)
argp.add_argument('--out', default='out.json')



def get_optimizer(t):
    if t == 'torch':
        return TorchOptimizer()
    elif t == 'LP':
        return LinProgOptimizer(True)
    elif t == 'MILP':
        return LinProgOptimizer(False)
    else:
        raise ValueError(t)

def fix_res(dat):
    f = Path(dat['dataset_path'])
    weights = pd.Series(dat['boost_weights'])
    if len(weights) ==0:
        return dat
    weights.index = pd.MultiIndex.from_tuples(list(map(eval, weights.index)))

    const = Optimizer.read_constraints(f)
    const = Optimizer.truncate_topk(const, dat['topk'])

    dat['total_violated'] = Optimizer.create_results(const, weights, dat['method_name'], pd.DataFrame(columns=['time', 'obj_val']))

    return dat



def main(args):
    if args.k < 1:
        raise ValueError(args.k)

    with open(args.input) as ifs, open(args.out, 'w') as ofs:
        for l in ifs:
            if len(l) <= 1:
                continue
            res = fix_res(json.loads(l))
            ofs.write(json.dumps(res)) 
            ofs.write('\n')


if __name__ == '__main__':
    main(argp.parse_args())
