import sys

from pydantic import NoneIsAllowedError

DATASET = sys.argv[1]

cnn_path = ''
cmv_path = ''
print('Loading data...')
if len(sys.argv) == 4:
    # './style_data_augmentation/sels/_cnn_out.txt'
    cnn_path = sys.argv[2]

    # './style_data_augmentation/sels/cmv_out.txt'
    cmv_path = sys.argv[3]
elif len(sys.argv) == 3:
    cnn_path = sys.argv[2]
else:
    raise NotImplementedError('This number of arguments is not supported.')

cmv_data = None
if cmv_path:
    with open(cmv_path, 'r') as f:
        cmv_data = f.readlines()

cnn_data = None
if cnn_path:  
    with open(cnn_path, 'r') as f:
        cnn_data = f.readlines()

print('Creating lookup...')
from load_data import get_persuasive_pairs_xml, load_cnn_dataset, load_cmv_dataset, load_imdb_dataset
_, persuasive_lookup = get_persuasive_pairs_xml()
_, imdb_lookup = load_imdb_dataset()
_, cnn_lookup = load_cnn_dataset()
_, cmv_lookup = load_cmv_dataset()

lookup = {**imdb_lookup, **cnn_lookup, **cmv_lookup}

datasets = {
    'cmv': cmv_data,
    'cnn': cnn_data
}

print('Creating new datasets...')
for name, dataset in datasets.items():
    print(name)
    if not dataset:
        continue
    with open(f'./style_data_augmentation/{DATASET}_{name}_dataset.txt', 'w') as f:
        # step size is 3 since 3 features
        for i in range(0, len(dataset), 3):
            sent, score = dataset[i].strip(), dataset[i+2].strip()
            try:
                prompt = lookup[dataset[i].strip()].strip()
            except KeyError:
                try:
                    prompt = lookup[dataset[i+1].strip()].strip()
                except KeyError:
                    continue

            if not len(prompt):
                prompt = lookup[dataset[i+1].strip()].strip()

            f.write(prompt + '\t' + score + '\t' + sent + '\n')
