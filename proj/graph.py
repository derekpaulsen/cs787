import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser


DATA_DIR = Path('./exp_res/')
# json data
MILP = DATA_DIR / 'MILP.json'
LP = DATA_DIR / 'LP.json'
TORCH = DATA_DIR / 'torch_fixed.json'
argp = ArgumentParser()
argp.add_argument('--show_graph', action='store_true')

TIME_LIMIT = 2700
DF = None

def process_time_series(row):

    s = pd.DataFrame(row['time_series']).astype(np.int32).set_index('time')['obj_val']
    s = s.groupby(s.index).last()
    s = s.loc[s.index < TIME_LIMIT]
    s[TIME_LIMIT] = row['total_violated']
    return s


def read_data(f):
    with open(f) as ifs:
        df = pd.DataFrame([json.loads(x) for x in ifs if len(x) > 1]).set_index('dataset')

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

def graph(milp, torch, show):
    print(milp.index)
    for ds in milp.index:
        ts = pd.concat([
            milp.at[ds, 'time_series'],
            torch.at[ds, 'time_series']
        ], axis=1).fillna(method='ffill').sort_index()
        print_stats(ts, ds)
        if show:
            ax = ts.plot()
            ax.set_title(ds)
            ax.set_ylim(bottom=0.0)
            ax.legend()
            plt.show()
        

def main(args):
    milp = read_data(MILP)
    torch = read_data(TORCH)
    graph(milp, torch, args.show_graph)


if __name__ == '__main__':
    main(argp.parse_args())




