import sys
sys.path.append('./persuasive_classifier')

import pandas as pd

from utils.load_data import get_memorable_pairs_imdb

import torch
from models.models import StyleClassifierScore
from transformers import BertTokenizer

model = StyleClassifierScore(n_classes=2)
model = torch.load("imdb_model.pth")

memorable_pairs = get_memorable_pairs_imdb('./persuasive_classifier/cornell_movie_quotes_corpus/moviequotes.memorable_nonmemorable_pairs.txt', include_metadata=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

inference_results = []
for pair in memorable_pairs:
    inputs = tokenizer(
        pair['sentence_a'],
        pair['sentence_b'],
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        return_token_type_ids=True,
        return_tensors='pt'
    )

    if inputs['input_ids'].shape[-1] > 512:
        for key, item in inputs.items():
            batch_size, tokens = item.shape
            new_item = torch.zeros(batch_size, 512)
            new_item = item[:, (tokens-512):]
            inputs[key] = new_item
            
    inputs = {key: item.to("cuda") for key, item in inputs.items()}

    outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
    outputs = float(outputs.detach().cpu().data[0])

    confidence = abs(outputs - 0.5) * 2
    result = pair
    result['confidence'] = confidence
    inference_results.append(result)

pd.DataFrame(inference_results).to_csv('inference_results.csv')

memorable_sents = set()
with open('imdb_full.txt','w') as f:
    for result in inference_results:
        memorable_sent = result['actual_mem_quote'].replace('\n','').strip()
        f.write(result['title'].replace('\n', '').strip() + '\t' + str(result['confidence']) + '\t' + memorable_sent + '\n')
        memorable_sents.add(memorable_sent)
    
with open('imdb_unique.txt', 'w') as f:
    for sent in memorable_sents:
        f.write(sent + '\n')





