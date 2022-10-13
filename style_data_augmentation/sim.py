import os
import numpy as np
import pandas as pd
import faiss
import sys

from load_data import load_cnn_dataset, load_cmv_dataset, get_persuasive_pairs_xml, load_imdb_dataset

home_dir = './'

k_mapping = {
    'cmv': 5,
    'cnn': 3
}

def create_index(database):
    d = database.shape[-1]               # dimension
    index = faiss.IndexFlatL2(d)   # build the index
    print(index.is_trained)
    index.add(database)            # add vectors to the index
    print(index.ntotal)
    return index

def get_sims(db_dataset, dataset_type, db_dataset_type):
    if dataset_type == 'cnn':
        query_dataset, _ = load_cnn_dataset()   
    elif dataset_type == 'cmv':
        query_dataset, _ = load_cmv_dataset()
    
    print('Loading embeddings...')
    xq = np.ascontiguousarray(pd.read_csv(os.path.join(home_dir, 'embs/' + dataset_type + '_embs.csv')).to_numpy()).astype('float32')
    xb = np.ascontiguousarray(pd.read_csv(os.path.join(home_dir, 'embs/' + db_dataset_type + '_embs.csv')).to_numpy()).astype('float32')
    
    print('Creating index...')
    index = create_index(xb)
    _, I = index.search(xq, k_mapping[dataset_type])     # actual search

    print('Writing to file')
    with open(os.path.join(home_dir, 'sims/' + dataset_type + '_' + db_dataset_type + '_sim.txt'), 'w') as f:
        for i, sent in enumerate(query_dataset):
            f.write(sent.replace('\n', ' ') + '\n')
            for idx in I[i]:
                f.write(db_dataset[idx].replace('\n', ' ') + '\n')

if __name__ == '__main__':
    
    db_dataset_type = 'imdb'
    if len(sys.argv) >= 2:
        db_dataset_type = sys.argv[1]

    if db_dataset_type == '16k':
        print('Loading persuasive pairs dataset...')
        sents, _ = get_persuasive_pairs_xml()
    elif db_dataset_type == 'imdb':
        print('Loading IMDB dataset...')
        sents, _ = load_imdb_dataset()

    dataset_types = [
       'cmv',
        'cnn'
    ]
    for dataset_type in dataset_types:
        get_sims(sents, dataset_type, db_dataset_type)

    
    
