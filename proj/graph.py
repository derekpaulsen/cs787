import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = Path('./exp_res')
# json data
MILP = DATA_DIR / 'fixed_MILP.json'
LP = DATA_DIR / 'LP.json'
TORCH = DATA_DIR / 'torch.json'


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


def graph(milp, torch):
    for ds in milp.index:
        ts = pd.concat([
            milp.at[ds, 'time_series'],
            torch.at[ds, 'time_series']
        ], axis=1).fillna(method='ffill').sort_index()
        print(ts)
        ax = ts.plot()
        ax.set_title(ds)
        ax.legend()
        plt.show()
    

def main():
    milp = read_data(MILP)
    torch = read_data(TORCH)
    graph(milp, torch)


if __name__ == '__main__':
    main()




