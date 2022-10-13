import re
import glob
import tarfile
import os.path
import json
from bz2 import BZ2File
from urllib import request
from io import BytesIO
import xml.etree.ElementTree as ET

from datasets import load_dataset
from transformers import BertTokenizer

def clean_text(text):
    text = text.strip().replace('\n', ' ').replace('\t', ' ')
    text = re.sub(' +', ' ', text)
    text = re.sub(r'\(http.*\)', '', text)
    text = re.sub(r'\\', '', text)
    text = text.replace('[', '').replace(']', '')
    return text

def get_persuasive_pairs_xml(directory: str = '../style_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/'):
    persuasive_pairs_df = []
    persuasive_pairs_lookup = {}

    for filename in glob.glob(directory + '*.xml'):
        root = ET.parse(filename).getroot()

        argument_pairs = [type_tag for type_tag in root.findall(
            'annotatedArgumentPair')]

        for argument_pair in argument_pairs:
            sentence_a = clean_text(argument_pair.find('arg1/text').text)
            sentence_b = clean_text(argument_pair.find('arg2/text').text)
            title = argument_pair.find('debateMetaData/title').text.strip()

            try:
                description = argument_pair.find('debateMetaData/description').text
            except AttributeError:
                description = title

            labels = [type_tag.find('value').text for type_tag in argument_pair.findall(
                'mTurkAssignments/mTurkAssignment')]
            label = max(labels, key=labels.count)
            label = int(label[-1]) if 'equal' not in label else 0
            if not label:
                continue
            
            # labels should be 0 and 1 if no equal arguments
            label -= 1
            row = sentence_a if not label else sentence_b
            persuasive_pairs_df.append(row)

            persuasive_pairs_lookup[sentence_a] = clean_text(title)
            persuasive_pairs_lookup[sentence_b] = clean_text(title)

    return list(dict.fromkeys(persuasive_pairs_df)), persuasive_pairs_lookup

def cleanup(cmv_post):
    lines = [line for line in cmv_post.splitlines()
             if not line.lstrip().startswith("&gt;")
             and not line.lstrip().startswith("____")
             and "edit" not in " ".join(line.lower().split()[:2])
            ]
    return list(filter(lambda x: len(x), (' '.join(lines)).split('. ')))

def include_sent(sent, blacklist_phrases=[' EST,']):
    for phrase in blacklist_phrases:
        if phrase in sent:
            return False

    return True

def load_cnn_dataset():
    print('Loading CNN/DM...')
    dataset = load_dataset("ccdv/cnn_dailymail", '3.0.0')
    highlights = list(dataset['train']['highlights'])
    dataset = list(dataset['train']['article'])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    full_dataset = []
    cnn_lookup = {}
    for i, d in enumerate(dataset):
        sents = d.split('. ')
        sents = list(map(lambda x: clean_text(x).replace('(CNN) -- ', ''), sents))
        sents = group_sentences(sents, tokenizer)
        
        for sent in sents:
            cnn_lookup[sent] = clean_text(highlights[i])
            if include_sent(sent):
                full_dataset.append(sent)
            
    return full_dataset, cnn_lookup

def download_cmv():
    fname = "cmv.tar.bz2"
    url = "https://chenhaot.com/data/cmv/" + fname

    # download if not exists
    if not os.path.isfile(fname):
        f = BytesIO()
        with request.urlopen(url) as resp, open(fname, 'wb') as f_disk:
            data = resp.read()
            f_disk.write(data)  # save to disk too
            f.write(data)
            f.seek(0)

def load_cmv_dataset():
    print('Loading CMV dataset...')
    fname = "./style_data_augmentation/cmv.tar.bz2"
    f = open(fname, 'rb')

    tar = tarfile.open(fileobj=f, mode="r")

    # Extract the file we are interested in

    train_fname = "op_task/train_op_data.jsonlist.bz2"
    test_fname = "op_task/heldout_op_data.jsonlist.bz2"

    train_bzlist = tar.extractfile(train_fname)

    # Deserialize the JSON list
    original_posts_train = [
        json.loads(line.decode('utf-8'))
        for line in BZ2File(train_bzlist)
    ]

    test_bzlist = tar.extractfile(test_fname)

    original_posts_test = [
        json.loads(line.decode('utf-8'))
        for line in BZ2File(test_bzlist)
    ]
    f.close()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    all_sents = []
    cmv_lookup = {}
    for post in original_posts_train + original_posts_test:
        sents = cleanup(post['selftext'])
        sents = list(map(lambda x: clean_text(x), sents))
        sents = group_sentences(sents, tokenizer)

        title = post['title'].replace('CMV: ', '')
        clean_title = clean_text(title)
        cmv_lookup[title] = clean_title

        for sent in sents:
            cmv_lookup[sent] = clean_title

        all_sents += list(map(lambda x: x.strip(), sents + [title])) 
    return all_sents, cmv_lookup

def load_imdb_dataset(imdb_path='./style_data_augmentation/moviequotes.memorable_quotes.txt'):
    print('Loading IMDB dataset...')
    memorable_pairs = []
    memorable_lookup = {}
    with open(imdb_path, encoding="latin-1") as f:
        lines = f.readlines()
        lines = list(map(lambda x: x.replace('\n', ''), lines))

    lines = list(filter(lambda a: a.strip() != '', lines))
    for title, memorable_quote, memorable_line in zip(lines[0::3], lines[1::3], lines[2::3]):
        memorable_lookup[title] = memorable_quote
        memorable_pairs.append(memorable_quote)

    return memorable_pairs, memorable_lookup

def group_sentences(sents, tokenizer, max_len=256, delim='. ', blacklist_phrases=None):
    sent_groups = []

    staged_sentence = ""
    cur_tokens = 0
    for sent in sents:
        if blacklist_phrases is not None and not include_sent(sent, blacklist_phrases=blacklist_phrases):
            continue
        elif not include_sent(sent):
            continue

        input_ids = tokenizer(
            sent,
            return_tensors='pt'
        )['input_ids']

        new_tokens = input_ids.shape[-1]
        if cur_tokens + new_tokens <= max_len:
            if len(staged_sentence):
                staged_sentence += delim
            staged_sentence += sent
            cur_tokens += new_tokens
        else:
            # send staged sentence off
            sent_groups.append(staged_sentence)
            staged_sentence = ""
            cur_tokens = 0
    
    if len(staged_sentence):
        sent_groups.append(staged_sentence)

    return sent_groups

if __name__ == '__main__':
    all_sents, cmv_lookup = load_cmv_dataset()
    sents, lookup = load_imdb_dataset()
            

