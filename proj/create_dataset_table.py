import pandas as pd
from pathlib import Path

files = Path('./data/').glob('*.parquet')

#'Walmart-Amazon' 

NAME_TO_ID = dict(
    zip(['Abt-Buy',
    'Amazon-Google', 
    'DBLP-GoogleScholar', 
    'uwhealth', 
    'electronics', 
    'Music'],

    range(10))
)

print(NAME_TO_ID)


def make_row(f):
    df = pd.read_parquet(f)

    return {
            'Dataset Name' : f.stem,
            'Number of Constraints' : len(df),
            'Number of Columns' : len(df.columns),
            'Percent Non-Zero' : df.ne(0).mean().mean() * 100
    }

table = pd.DataFrame(list(map(make_row, files)))

print(table)

table = table.loc[table['Dataset Name'].isin(NAME_TO_ID)]
table.index = table['Dataset Name'].apply(NAME_TO_ID.__getitem__).values
table = table.sort_index()

table['Dataset Name'] = table['Dataset Name'].apply(lambda x : f'D_{NAME_TO_ID[x]}')
print(table)

print(table.to_latex(index=False, column_format='|l|r|r|r|'))


