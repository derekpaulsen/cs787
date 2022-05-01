import json 
import pandas as pd

res = list(reversed(list(map(json.loads, open('./MILP.json')))))

for i in range(1, len(res)):
    prev = pd.DataFrame(res[i-1]['time_series'])
    curr = pd.DataFrame(res[i]['time_series'])
    res[i-1]['time_series'] = prev.iloc[len(curr):].to_dict(orient='list')

with open('fixed_MILP.json', 'w') as ofs:
    for r in reversed(res):
        ofs.write(json.dumps(r))
        ofs.write('\n')






