import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from argparse import ArgumentParser
from utils import NAME_TO_ID
from optimizer import Optimizer
#matplotlib.rc('text', usetex=True)
#matplotlib.rc('font', family='serif')

TIME_LIMIT = Optimizer.TIMEOUT
DATA_DIR = Path('./exp_res/')
# json data
MILP = DATA_DIR / 'MILP.json'
LP = DATA_DIR / 'LP.json'
TORCH = DATA_DIR / 'torch.json'
argp = ArgumentParser()
argp.add_argument('--show_graph', action='store_true')

DF = None

def process_time_series(row):

    s = pd.DataFrame(row['time_series']).astype(np.int32).set_index('time')['obj_val']
    s = s.groupby(s.index).last()
    time = min(TIME_LIMIT, int(row['opt_time']))
    s = s.loc[s.index < time]
    s[time] = row['total_violated']

    return s / row['num_constraints'] * 100


def read_data(f):
    with open(f) as ifs:
        df = pd.DataFrame([json.loads(x) for x in ifs if len(x) > 1])

    df['dataset_id'] = df['dataset'].apply(lambda x : NAME_TO_ID.get(x))
    df = df.loc[df['dataset_id'].notnull()].set_index('dataset_id')

    method_name = df['method_name'].iloc[0]
    df['time_series'] = [process_time_series(x[1]).rename(method_name) for x in df.iterrows()]

    return df

def print_stats(ts, ds):
    if 'MILP' in ts.columns and ts['MILP'].lt(ts['torch']).any():
        t_overtake = ts['MILP'].lt(ts['torch']).idxmax()
    else:
        t_overtake = np.nan

    print(ds)
    print(f'time for MILP to overtake torch : {t_overtake}')
    print(f'time for MILP to reach best value: {ts.MILP.idxmin()}')
    print(f'time for torch to reach best value: {ts.torch.idxmin()}')
    print(ts)


def iter_time_series(milp, torch):
    for name, i in NAME_TO_ID.items():
        if i not in torch.index or i not in milp.index:
            continue
        ts = pd.concat([
            milp.at[i, 'time_series'],
            torch.at[i, 'time_series']
        ], axis=1).sort_index()
        idx = ts['MILP'].last_valid_index()
        ts['MILP'].loc[:idx] = ts['MILP'].loc[:idx].fillna(method='ffill')
        ts['torch']= ts['torch'].fillna(method='ffill')

        yield name, ts


def graph(milp, torch, show):
    print(milp.index)
    for name, ts in iter_time_series(milp, torch):
        print_stats(ts, name)
        if show:
            ax = ts.plot()
            ax.set_title(name)
            #ax.set_ylim(bottom=0.0)
            ax.legend()
            plt.show()

def graph_six(milp, torch):
    time_series = list(iter_time_series(milp, torch))
    fig, axes = plt.subplots(3, 2, figsize=(12,8))
    
    for ax, ts_data, i in zip(axes.flatten(), time_series, range(len(time_series))):
        name, ts = ts_data
        ts.plot(ax=ax)
        ax.set_title(f'$D_{i}$')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Percent of Constraints Violated')
        ax.legend()
    
    fig.tight_layout()
    fig.savefig('./paper/time_series.png')

    plt.show()

        

def main(args):
    milp = read_data(MILP)
    torch = read_data(TORCH)
    if args.show_graph:
        graph_six(milp, torch)
    else:
        graph(milp, torch, args.show_graph)


if __name__ == '__main__':
    main(argp.parse_args())




