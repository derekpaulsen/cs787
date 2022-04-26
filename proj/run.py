from argparse import ArgumentParser
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

argp.add_argument('--method', required=True, choices=['torch', 'MILP', 'LP'])
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

def run(opt, f, k):
    log.info(f'running {f}')
    const = Optimizer.read_constraints(f)
    const = Optimizer.truncate_topk(const, k)
    w = opt.optimize(const)

    res = opt.results.copy()
    res['dataset'] = f.stem
    res['dataset_path'] = str(f.absolute())
    res['time_ran'] = str(datetime.now())
    res['num_constraints'] = len(const)
    res['topk'] = k
    log.info(f'\n{pformat(res)}')

    return res


def main(args):
    if args.k < 1:
        raise ValueError(args.k)

    with open(args.out, 'a') as ofs:
        for ds in datasets:
            if 'Music' in ds.name:
                opt = get_optimizer(args.method)
                res = run(opt, ds, args.k)
                ofs.write(json.dumps(res)) 
                ofs.write('\n')


if __name__ == '__main__':
    main(argp.parse_args())
