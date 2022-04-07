from pathlib import Path
import pandas as pd
from optimizer import Optimizer


data = Path('./data')


for f in data.glob('*.parquet'):
    df = Optimizer.read_constraints(f)
    print(f.name)
    print(df)
