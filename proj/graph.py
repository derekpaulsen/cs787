import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


DATA_DIR = Path('./exp_res/k=50')
# json data
MILP = DATA_DIR / 'MILP.json'
LP = DATA_DIR / 'LP.json'
TORCH = DATA_DIR / 'torch.json'
argp = ArgumentParser()
argp.add_argument('--show_graph', action='store_true')


DF = None

def process_time_series(x):

    s = pd.DataFrame(x).astype(np.int32).set_index('time')['obj_val']
    s = s.groupby(s.index).last()
    return s


def read_data(f):
    with open(f) as ifs:
        df = pd.DataFrame(list(map(json.loads,ifs))).set_index('dataset')

    method_name = df['method_name'].iloc[0]
    df['time_series'] = [process_time_series(x).rename(method_name) for x in df['time_series'].values]

    return df

def print_stats(df, ds):

    t_overtake = df['MILP'].lt(df['torch']).idxmax()
    print(ds)
    print(f'time for MILP to overtake torch : {t_overtake}')
    print(f'time for MILP to reach best value: {df.MILP.idxmin()}')
    print(f'time for torch to reach best value: {df.torch.idxmin()}')
    print()

def graph(milp, torch, show):
    for ds in milp.index:
        ts = pd.concat([
            milp.at[ds, 'time_series'],
            torch.at[ds, 'time_series']
        ], axis=1).fillna(method='ffill').sort_index()
        print_stats(ts, ds)
        if show:
            ax = ts.plot()
            ax.set_title(ds)
            ax.legend()
            plt.show()
        

def main(args):
    milp = read_data(MILP)
    torch = read_data(TORCH)
    graph(milp, torch, args.show_graph)


if __name__ == '__main__':
    main(argp.parse_args())




