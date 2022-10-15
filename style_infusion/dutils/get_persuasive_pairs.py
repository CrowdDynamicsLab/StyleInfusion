import glob
import random
import xml.etree.ElementTree as ET
from collections import OrderedDict

import torch
import torch.nn as nn
import transformers
from transformers import BertModel, BertTokenizer

class StyleClassifierScore(nn.Module):
    def __init__(self, model_name='bert-base-uncased', hidden_dim=768, dropout=0.2, n_classes=3):
        super(StyleClassifierScore, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_batch):
        outputs = self.bert(**input_batch)
        output = self.dropout(outputs.pooler_output)
        output = self.linear(output)
        output = self.sigmoid(output.flatten())
        # Since 0.5 corresponds to equal confidence, we use the following to get confidence on a scale of 0 to 1
        output = 2 * torch.abs(0.5 - output)
        return output

def get_persuasive_pairs_xml(persuasive_classifier, tokenizer, directory: str = './16k_persuasiveness/data/UKPConvArg1Strict-XML/'):
    '''
    Extracts and compiles the 16k persuasiveness pairs from a directory containing XML files

    Args:
        directory (str): directory to 16k persuasive pairs folder with XML files
        include_metadata (bool): whether to include filename in data
    Returns:
        a list of dictionaries with both sentences and the label
    '''
    persuasive_pairs_df = []
    sentence_description_lookup = {}

    for filename in glob.glob(directory + '*.xml'):
        root = ET.parse(filename).getroot()

        argument_pairs = [type_tag for type_tag in root.findall(
            'annotatedArgumentPair')]

        for argument_pair in argument_pairs:
            sentence_a = argument_pair.find('arg1/text').text.strip().replace('\n', ' ')
            sentence_b = argument_pair.find('arg2/text').text.strip().replace('\n', ' ')
            title = argument_pair.find('debateMetaData/title').text.strip()
            try:
                description = argument_pair.find('debateMetaData/description').text.strip().replace('\n', ' ')
            except AttributeError:
                description = title
            
            description = title if description is None else description

            labels = [type_tag.find('value').text for type_tag in argument_pair.findall(
                'mTurkAssignments/mTurkAssignment')]
            labels = filter(lambda x: x != 'equal', labels) 
            labels = list(map(lambda x: int(x[-1]) - 1, labels))
            label = max(labels, key=labels.count)

            inputs = tokenizer(
                sentence_a,
                sentence_b,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                return_token_type_ids=True,
                return_tensors='pt'
            )

            inputs = {key: item.to("cuda") for key, item in inputs.items()}

            persuasive_score = persuasive_classifier(inputs)
            persuasive_score = persuasive_score.cpu().detach().data
            
            if (not label and persuasive_score < 0.5) or (label == 1 and persuasive_score >= 0.5):
                persuasive_score = abs(persuasive_score)
            else:
                persuasive_score = labels.count(label)/len(labels)

            persuasive_sentence = sentence_a if not label else sentence_b

            row = {
                    'persuasive_sentence': persuasive_sentence,
                    # 'description': description,
                    'persuasive_score': persuasive_score
            }

            sentence_description_lookup[persuasive_sentence] = description
            persuasive_pairs_df.append(row)

    return persuasive_pairs_df, sentence_description_lookup

def write_persuasive_df_to_txt(persuasive_pairs_df, sentence_description_lookup, txt_filename = './16k_pairs', train_test_split=0.8):

    # remove duplicate entries and average the scores
    sentence_to_scores = {}
    for row in persuasive_pairs_df:
        persuasive_sentence = row['persuasive_sentence']
        score = float(row['persuasive_score'])
        if persuasive_sentence in sentence_to_scores:
            sentence_to_scores[persuasive_sentence].append(score)
        else:
            sentence_to_scores[persuasive_sentence] = [score]

    persuasive_pairs_df = [{key: sum(value)/len(value)} for key, value in sentence_to_scores.items()]

    if train_test_split == 1.0:
        with open(f'{txt_filename}_full.txt', 'w') as f:
            for row in persuasive_pairs_df:
                persuasive_sentence, score = list(row.keys())[0], list(row.values())[0]
                description = sentence_description_lookup[persuasive_sentence]
                data_str = description + '\t' + str(score) + '\t' + persuasive_sentence + '\n'
                f.write(data_str)
    else:
        # train test split
        total_length = len(persuasive_pairs_df)
        random.shuffle(persuasive_pairs_df)
        split = int(train_test_split * total_length)
        train_dataset = persuasive_pairs_df[:split]
        test_dataset = persuasive_pairs_df[split:]

        for dataset, set_name in zip([train_dataset, test_dataset], ['train', 'test']):
            with open(f'{txt_filename}_{set_name}.txt', 'w') as f:
                for row in dataset:
                    persuasive_sentence, score = list(row.keys())[0], list(row.values())[0]
                    description = sentence_description_lookup[persuasive_sentence]
                    data_str = description + '\t' + str(score) + '\t' + persuasive_sentence + '\n'
                    f.write(data_str)


if __name__ == '__main__':
    persuasive_classifier = StyleClassifierScore()

    sd = torch.load('persuasive_model.pth')
    new_sd = OrderedDict()

    for key, value in sd.items():
        new_key = key.replace('module.', '')
        new_sd[new_key] = value
    sd = new_sd

    persuasive_classifier.load_state_dict(sd)
    persuasive_classifier = persuasive_classifier.cuda()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    persuasive_pairs_df, sentence_description_lookup = get_persuasive_pairs_xml(persuasive_classifier, tokenizer, '../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/')
    write_persuasive_df_to_txt(persuasive_pairs_df, sentence_description_lookup, train_test_split=1.0)