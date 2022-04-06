from pathlib import Path
import pandas as pd
from optimizer import Optimizer


data = Path('./data')


for f in data.glob('*.parquet'):
    df = pd.read_parquet(f)
    print(f.name)
    s = df.groupby(level=0).count()
    print(s)
