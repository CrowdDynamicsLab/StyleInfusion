import argparse
from collections import OrderedDict
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


from data_feature_utils import get_features_dicts, update_dict_list
from utils.load_data import PersuasivePairsDataset, get_persuasive_pairs_xml, get_persuasive_pairs_lookup
from style_classifier.models import StyleClassifier, StyleClassifierScore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(args, state_dict=False):
    '''
    Loads in persuasive classifier model. 
    Set state_dict option to true if the checkpoint stores only the state dict.
    '''
    model = StyleClassifierScore(args.model_name_or_path, dropout=args.dropout, n_classes=int(
            2 if args.exclude_equal_arguments else 3))

    if state_dict:
        sd = torch.load(args.trained_model_path).state_dict()
        new_sd = OrderedDict()
        
        for key, value in sd.items():
            new_key = key.replace('module.', '')
            new_sd[new_key] = value
        sd = new_sd

        model.load_state_dict(sd)
    else:
        model = torch.load(args.trained_model_path, map_location=device)

    if not args.no_cuda:
        model = model.to(device)
    
    return model


def run_inference_persuasive_pairs(data_pairs, model, tokenizer):
    '''
    Runs inference on the data pairs using the persuasiveness classifier.
    Returns a list of dictionaries with the inference results.
    '''
    inference_results = []

    dataset = PersuasivePairsDataset(data_pairs, tokenizer, include_metadata=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=0)
    for _, data in enumerate(dataloader):
        
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(
            device, dtype=torch.long)

        label = data['label'].to(device, dtype=torch.long)
        outputs = model(ids, mask, token_type_ids)

        sentence_a_ids = data['a_id']
        sentence_b_ids = data['b_id']

        for i in range(min(args.batch_size, len(sentence_a_ids), len(sentence_b_ids))):
            d = {
                'a_id': sentence_a_ids[i],
                'b_id': sentence_b_ids[i],
                'score': outputs[i].cpu().detach().item(),
                'label': label[i].item()
            }

            inference_results.append(d)
        
    return inference_results


class GenerationPairsDataset(Dataset):
    """Generation pairs dataset."""

    def __init__(
        self,
        sentence_a_corpus,
        sentence_b_corpus,
        tokenizer,
        max_len: int = 512,
    ):
        self.sentence_a_corpus = sentence_a_corpus
        self.sentence_b_corpus = sentence_b_corpus

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentence_a_corpus)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.data)

        sentence_a = self.sentence_a_corpus[idx]
        sentence_b = self.sentence_b_corpus[idx]

        inputs = self.tokenizer(
            sentence_a,
            sentence_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        item = {
            'ids': torch.tensor(inputs['input_ids']),
            'mask': torch.tensor(inputs['attention_mask']),
            'token_type_ids': torch.tensor(inputs["token_type_ids"]),
            'sentence_a': sentence_a,
            'sentence_b': sentence_b
        }

        return item

def run_inference_pairs(args, sentence_a_corpus, sentence_b_corpus, model, tokenizer):
    inference_results = []

    dataset = GenerationPairsDataset(sentence_a_corpus, sentence_b_corpus, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    for _, data in enumerate(dataloader):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)

        outputs = model(ids, mask, token_type_ids)

        for i in range(min(args.batch_size, data['ids'].shape[0])):
            d = {
                'sentence_a': data['sentence_a'][i],
                'sentence_b': data['sentence_b'][i],
                'score': outputs[i].cpu().detach().item(),
            }

            inference_results.append(d)

    return inference_results


def get_features_dict(persuasive_pairs, mtcg_verbs_path, stanza_config_path):
    '''
    Gets all linguistic features from `data_feature_utils.py`.
    You can also call this directly from that file.
    '''
    texts, ids, features_dict = [], [], []
    for row in persuasive_pairs:
        if row['a_id'] not in ids:
            texts.append(row['sentence_a'])
            ids.append(row['a_id'])
            features_dict.append({'id': row['a_id']})

        if row['b_id'] not in ids:
            texts.append(row['sentence_b'])
            ids.append(row['b_id'])
            features_dict.append({'id': row['b_id']})

    update_dict_list(features_dict, get_features_dicts(texts, mtcg_verbs_path=mtcg_verbs_path, stanza_config_path=stanza_config_path))
    return features_dict

def get_controls(persuasive_pairs):
    '''
    Retrieves a DataFrame of controls for the regression.
    Add as many controls as you can for your own use case.
    '''
    persuasive_sents = {}
    for pair in persuasive_pairs:
        persuasive_sents[pair['a_id']] = (pair['sentence_a'], pair['filename'])
        persuasive_sents[pair['b_id']] = (pair['sentence_b'], pair['filename'])
    
    filename_to_id = {}
    controls_list = []
    for key, value in persuasive_sents.items():
        sent, filename = value

        # control for the filename
        if filename not in filename_to_id:
            filename_to_id[filename] = len(filename_to_id)

        controls = {'id': key, 'filename': filename_to_id[filename]}
        controls_list.append(controls)

    return pd.DataFrame(controls_list)

def standardize(x):
    a = x.mean()
    s = x.std()
    return((x-a)/s)

def reorder_result_columns(df, save_file="reordered_results.csv"):
    with open('correlations/imdb_correlations.txt') as f:
        lines = f.readlines()
    scores = []
    for line1, line2 in zip(lines[::2], lines[1::2]):
        scores.append((' '.join(line1.split(',')[0].split('_')).title(), float(line2.replace('array(', '').replace(', dtype=float32)\n', ''))))

    n=6
    sorted_scores = list(sorted(scores, key=lambda x:x[1]))
    new_cols = list(map(lambda x: x.lower().replace(' ', '_'), [s[0] for s in sorted_scores]))

    new_cols = ['model'] + new_cols
    df.columns = list(map(lambda x: x.lower(), list(df.columns)))

    d2 = set(new_cols).difference(set(df.columns))
    new_cols = [c for c in new_cols if c not in d2]
    df = df[new_cols]
    df = df.sort_values(by='model')
    df.to_csv(save_file)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str,
                        default='bert-base-uncased', help="Backbone model name/path")
    parser.add_argument("--trained_model_path", type=str, default="./persuasive_model.pth",
                        help="Already trained persuasive classifier path.")
    parser.add_argument("--exclude_equal_arguments", action="store_true",
                        help="Whether to exclude the equal class (0).")
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="Dropout for fully-connected layers")

    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--inference_path", type=str,
                    help="CSV to load inference from.")

    parser.add_argument("--features_path", type=str,
                    help="CSV to load features from.")
    parser.add_argument("--mtcg_verbs_path", type=str, default='mtcg_verbs.csv',
                    help="Path to csv containing mtcg verb classifications.")
    parser.add_argument("--stanza_config_path", type=str, default='stanza_config/stanza_config.yaml',
                    help="Path to Stanza config file.")

    parser.add_argument("--mode", type=str, default='gen', help='Whether to calculate features over \
        generations (with inference scores) or to calculate correlations from generations')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    if not args.trained_model_path and not args.inference_path:
        raise Exception('You need to have a trained model to do inference!')
    
    if args.mode == 'data':

        persuasive_pairs = get_persuasive_pairs_xml(include_metadata=True, exclude_equal_arguments=args.exclude_equal_arguments)
        sent_to_id, _ = get_persuasive_pairs_lookup(persuasive_pairs)

        # loads or generates the inference df
        inference_df = None
        if args.inference_path:
            inference_df = pd.read_csv(args.inference_path) 
        else:
            model = load_model(args, state_dict=True)
            tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
            inference_results = run_inference_persuasive_pairs(persuasive_pairs, model, tokenizer)
            inference_df = pd.DataFrame(inference_results).fillna(0)
            inference_df.to_csv('inference.csv', index=False)

        # get average scores for each sentence in 16k pairs
        # this is used as the dependent variable for the regression
        sentence_to_scores = {}
        for i, row in inference_df.iterrows():
            p_id = row['a_id'] if not int(row['label']) else row['b_id']
            score = float(row['score'])
            if p_id in sentence_to_scores:
                sentence_to_scores[p_id].append(score)
            else:
                sentence_to_scores[p_id] = [score]
        avg_score_per_sentence = [{'id': key, 'score': sum(value)/len(value)} for key, value in sentence_to_scores.items()]
        avg_score_per_sentence = pd.DataFrame(avg_score_per_sentence)

        # load in features (or calculate them)
        features_df = None
        if args.features_path:
            features_df = pd.read_csv(args.features_path) 
            if features_df.columns[0] != 'id':
                features_df.drop(columns=features_df.columns[0], axis=1, inplace=True)
        else:
            features_dict = get_features_dict(persuasive_pairs, args.mtcg_verbs_path, args.stanza_config_path) 
            features_df = pd.DataFrame(features_dict).fillna(0)
            features_df.to_csv(args.features_dir, index=False)

    elif args.mode == 'gen':
        print('Loading in generations....')
        generations = {}
        for filename in glob.glob('./style-infusion/generations/*.txt'):
            with open(filename) as f:
                generations[filename.split('/')[-1].replace('.txt','')] = f.readlines()

        print('Loading in features....')
        features = {}
        for filename in glob.glob('features/*.csv'):
            df = pd.read_csv(filename, index_col=0)
            features[filename.split('/')[-1].replace('.csv','').replace('_features','')] = df

        print('Loading in inferences....')
        inferences = {}
        for filename in glob.glob('inferences/*.csv'):
            df = pd.read_csv(filename, index_col=0)
            inferences[filename.split('/')[-1].replace('.csv','')] = df

        final_exp_vals = []
        # model = load_model(args, state_dict=True)
        model = None
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
        for key1, item1 in generations.items():
            for key2, item2 in generations.items():
                if key1 == key2 or key1 not in features or key2 not in features:
                    continue
                print(f'Doing {key1} + {key2}')
                # lookup features for generations before calculating otherwise get features
                features1 = features[key1].fillna(0) if key1 in features else get_features_dicts(item1, args.mtcg_verbs_path, args.stanza_config_path).fillna(0)
                features2 = features[key2].fillna(0) if key2 in features else get_features_dicts(item2, args.mtcg_verbs_path, args.stanza_config_path).fillna(0)

                inference_df = None
                if key1 + '+' + key2 in inferences:
                    # lookup inference for generations before calculating
                    inference_df = inferences[key1 + '+' + key2]
                else:
                    if model is None:
                        model = StyleClassifierScore()
                        model = torch.load(args.trained_model_path)
                    print('Running inference...')
                    # inference on generations
                    inference = run_inference_pairs(args, item1, item2, model, tokenizer)
                    inference_df = pd.DataFrame(inference).fillna(0)
                    inference_df.to_csv('inferences/' + key1 + '+' + key2 + '.csv', index=False)

                # compute expected value for feature
                # inference = [{'score'}, {}, {}]
                # features = [{sentence, features ....}]
                exp_vals = {'model': key1 + '+' + key2}
                for col in features1.columns:
                    if col in ['id', 'text'] or col not in features2.columns:
                        continue

                    # standardizing 
                    feat_col1 = standardize(features1[col])
                    feat_col2 = standardize(features2[col])

                    numerator = 0
                    denominator = 0
                    for i in range(len(features1)):

                        # argument 1 - argument 2
                        diff = feat_col1[i] - feat_col2[i]

                        # score is from 0 to 1
                        # if argument 1 wins, score is 0
                        # thus likelihood of argument 1 winning should be 0.5-score

                        numerator += diff * (0.5 - inference_df['score'][i])
                        denominator += (0.5 - inference_df['score'][i])

                    exp_val = numerator / denominator
                    exp_vals[col] = exp_val

                final_exp_vals.append(exp_vals)

        results_df = pd.DataFrame(final_exp_vals)
        reorder_result_columns(results_df)
                        