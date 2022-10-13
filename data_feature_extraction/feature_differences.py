import os
import sys
import glob
import pandas as pd
from scipy.stats import ttest_ind

if not os.path.exists('./correlations'):
    os.mkdir('./correlations')

if not os.path.exists('./features'):
    os.mkdir('./features')

DATASET = 'imdb' # set to 16k if persuasive pairs/imdb otherwise
if len(sys.argv) >= 2:
    DATASET = sys.argv[1]

if not os.path.exists(f'./features/{DATASET}'):
    os.mkdir(f'./features/{DATASET}')

def reorder_result_columns(df, save_file=f"{DATASET}_diffs.csv"):
    with open(f'correlations/{DATASET}_correlations.txt') as f:
        lines = f.readlines()
    scores = []
    for line1, line2 in zip(lines[::2], lines[1::2]):
        scores.append((' '.join(line1.split(',')[0].split('_')).title(), float(line2.replace('array(', '').replace(', dtype=float32)\n', ''))))

    n=6
    sorted_scores = list(sorted(scores, key=lambda x:x[1]))
    new_cols = list(map(lambda x: x.lower().replace(' ', '_'), [s[0] for s in sorted_scores]))

    new_cols = ['model1', 'model2'] + new_cols
    df.columns = list(map(lambda x: x.lower(), list(df.columns)))

    d2 = set(new_cols).difference(set(df.columns))
    new_cols = [c for c in new_cols if c not in d2]
    df = df[new_cols]
    df = df.sort_values(by=['model1','model2'])
    df.to_csv(save_file)


dfs = {}
for filename in glob.glob(f'features/{DATASET}/*.csv'):
    if DATASET =='imdb' and DATASET not in filename:
        # CHANGE THIS IF NOT USING IMDB DATASET
        continue

    df = pd.read_csv(filename, index_col=0)
    if 'id' in df.columns and 'text' in df.columns:
        dfs[filename.replace('.csv','')] = df.drop(['id','text'], axis=1).fillna(0)
    else:
        dfs[filename.replace('.csv','')] = df.fillna(0)    
diffs = []
for key1, val1 in dfs.items():
    for key2, val2 in dfs.items():
        if key1 == key2:
            continue

        diff_val = val1.subtract(val2)[val1.columns].mean()

        diff = {'model1': key1, 'model2': key2}
        for col in val1.columns:
            if col not in val2.columns or col in ['id', 'text']:
                continue

            ttest_result = ttest_ind(val1[col], val2[col])
            diff[col] = ttest_result.pvalue * (-1 if float(diff_val[col]) < 0 else 1)
        diffs.append(diff)

full_diffs = pd.DataFrame(diffs)
reorder_result_columns(full_diffs)