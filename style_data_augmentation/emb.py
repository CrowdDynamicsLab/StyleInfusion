#@title Load the Universal Sentence Encoder's TF Hub module
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd

from load_data import load_cmv_dataset, get_persuasive_pairs_xml, load_cnn_dataset, load_imdb_dataset

def embed(input):
  return model(input)

def get_cmv_embs(save_dir = 'embs/cmv_embs.csv'):
    '''
    Stores embeddings of the /r/changemyview dataset
    '''
    all_sents, _ = load_cmv_dataset()
    dataset_embeddings = embed(all_sents)
    df = pd.DataFrame(np.array(dataset_embeddings))
    df.to_csv(save_dir, index=False)

def store_cnn_dm_embs(save_dir = 'embs/cnn_embs.csv'):
    '''
    Stores embeddings of the CNN/DM dataset
    '''
    full_dataset, _ = load_cnn_dataset()
    dataset_len = len(full_dataset)

    N = 100
    chunk_len = int(dataset_len / N)

    total_len = 0
    for i in range(N):
        if i == N - 1:
            partial_dataset = full_dataset[i * chunk_len :]
        else:
            partial_dataset = full_dataset[i * chunk_len : (i + 1) * chunk_len]

        dataset_embeddings = embed(partial_dataset)
        df = pd.DataFrame(np.array(dataset_embeddings))
        total_len += len(df)
        df.to_csv(save_dir, mode='a', index=False)

    assert dataset_len == total_len

def store_16k_persuasiveness_embs(save_dir = 'embs/16k_embs.csv'):
    '''
    Stores embeddings of the 16k Persuasive Pairs dataset
    '''
    persuasive_sentences, _ = get_persuasive_pairs_xml()
    embeddings = embed(persuasive_sentences)
    df = pd.DataFrame(np.array(embeddings))
    df.to_csv(save_dir, index=False)

def store_imdb_embs(save_dir = 'imdb_embs.csv'):
    '''
    Stores embeddings of the IMDB dataset
    '''
    sents, _ = load_imdb_dataset()
    embeddings = embed(sents)
    df = pd.DataFrame(np.array(embeddings))
    df.to_csv(save_dir, index=False)

if __name__ == '__main__':
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
    model = hub.load(module_url)
    print ("module %s loaded" % module_url)

    get_cmv_embs()
    store_cnn_dm_embs()
    store_16k_persuasiveness_embs()
    store_imdb_embs()


