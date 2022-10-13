import glob
import torch
import random
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET


class PersuasivePairsDataset(Dataset):
    """Persuasive pairs dataset."""

    def __init__(
        self,
        persuasive_pairs_df,
        tokenizer,
        max_len: int = 512,
        columns: dict = {'label': 'label', 'sentence_a': 'sentence_a', 'sentence_b': 'sentence_b',
                         'confidence': 'confidence','filename': 'filename', 'a_id': 'a_id', 'b_id': 'b_id'},
        include_metadata: bool = False
    ):
        """
        Args:
            persuasive_pairs_df (pd.DataFrame): dataframe of persuasive pairs
            columns (dict): columns with default column names paired with custom ones
        """
        self.persuasive_pairs_df = persuasive_pairs_df
        self.columns = columns

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.include_metadata = include_metadata

    def __len__(self):
        return len(self.persuasive_pairs_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.data)

        data_idx = self.persuasive_pairs_df[idx]
        label = data_idx[self.columns['label']]
        sentence_a = data_idx[self.columns['sentence_a']]
        sentence_b = data_idx[self.columns['sentence_b']]
        confidence = data_idx[self.columns['confidence']]

        inputs = self.tokenizer(
            sentence_a,
            sentence_b,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )

        item = {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            'label': torch.tensor(int(label), dtype=torch.long),
            'confidence': torch.tensor(float(confidence), dtype=torch.float)
        }

        if self.include_metadata:
            metadata = {
                'filename': data_idx[self.columns['filename']],
                'a_id': data_idx[self.columns['a_id']],
                'b_id': data_idx[self.columns['b_id']],
            }
            item.update(metadata)

        return item


class SiamesePersuasivePairsDataset(Dataset):
    """Persuasive pairs dataset for the Siamese model."""

    def __init__(
        self,
        persuasive_pairs_df,
        columns: dict = {'label': 'label',
                         'sentence_a': 'sentence_a', 'sentence_b': 'sentence_b', 'confidence': 'confidence'},
    ):
        """
        Args:
            persuasive_pairs_df (pd.DataFrame): dataframe of persuasive pairs
            columns (dict): columns with default column names paired with custom ones
        """
        self.persuasive_pairs_df = persuasive_pairs_df
        self.columns = columns

    def __len__(self):
        return len(self.persuasive_pairs_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx.data)

        data_idx = self.persuasive_pairs_df[idx]
        label = data_idx[self.columns['label']]
        sentence_a = data_idx[self.columns['sentence_a']]
        sentence_b = data_idx[self.columns['sentence_b']]
        confidence = data_idx[self.columns['confidence']]

        return {
            'texts': [sentence_a, sentence_b],
            'label': int(label),
            'confidence': torch.tensor(float(confidence), dtype=torch.float)
        }


def get_persuasive_pairs_xml(directory: str = '../16k_persuasiveness/data/UKPConvArg1Strict-XML/', include_metadata: bool = False, exclude_equal_arguments: bool = False):
    '''
    Extracts and compiles the 16k persuasiveness pairs from a directory containing XML files

    Args:
        directory (str): directory to 16k persuasive pairs folder with XML files
        include_metadata (bool): whether to include filename in data
        exclude_equal_arguments (bool): whether to exclude the 0 label indicating the arguments are equally persuasive
    Returns:
        a list of dictionaries with both sentences and the label
    '''
    persuasive_pairs_df = []

    for filename in glob.glob(directory + '*.xml'):
        root = ET.parse(filename).getroot()

        argument_pairs = [type_tag for type_tag in root.findall(
            'annotatedArgumentPair')]

        for argument_pair in argument_pairs:
            sentence_a = argument_pair.find('arg1/text').text
            sentence_b = argument_pair.find('arg2/text').text

            labels = [type_tag.find('value').text for type_tag in argument_pair.findall(
                'mTurkAssignments/mTurkAssignment')]
            label = max(labels, key=labels.count)
            confidence = labels.count(label)/len(labels)
            label = int(label[-1]) if 'equal' not in label else 0
            if not label and exclude_equal_arguments:
                continue
            elif exclude_equal_arguments:
                # labels should be 0 and 1 if no equal arguments
                # if there are equal arguments, labels are 0 (equal), 1 and 2
                label -= 1

            row = {'label': label, 'sentence_a': sentence_a,
                   'sentence_b': sentence_b, 'confidence': confidence}

            if include_metadata:
                row['filename'] = filename
                row['a_id'] = argument_pair.find('arg1/id').text
                row['b_id'] = argument_pair.find('arg2/id').text
            persuasive_pairs_df.append(row)

    return persuasive_pairs_df

def get_persuasive_pairs_lookup(dataset):
    sent_to_id = {}
    id_to_sent = {}

    for row in dataset:
        sent_to_id[row['sentence_a']] = row['a_id']
        id_to_sent[row['a_id']] = row['sentence_a']
        sent_to_id[row['sentence_b']] = row['b_id']
        id_to_sent[row['b_id']] = row['sentence_b']

    return sent_to_id, id_to_sent

def get_memorable_pairs_imdb(path_to_data: str, include_metadata: bool = False, exclude_equal_arguments=None):
    '''
    Extracts and compiles the IMDB memorability dataset from a file

    Args:
        directory (str): directory to IMDB memorability dataset file
        include_metadata (bool): whether to include movie title/line numbers in data
        exclude_equal_arguments (bool): whether to include the 0 label indicating the quotes are equally memorable
    Returns:
        a list of dictionaries with both sentences and the label
    '''
    memorable_pairs = []
    with open(path_to_data, encoding="latin-1") as f:
        lines = f.readlines()

    lines = list(filter(lambda a: a.strip() != '', lines))
    for title, memorable_quote, memorable_line, non_memorable_line in zip(lines[0::4], lines[1::4], lines[2::4], lines[3::4]):
        mem_quote = ' '.join(memorable_line.split()[1:]).strip()
        non_mem_quote = ' '.join(non_memorable_line.split()[1:]).strip()
        sentence_a, sentence_b, label = None, None, None

        if random.random() < 0.5:
            sentence_a = mem_quote
            sentence_b = non_mem_quote
            label = 0
        else:
            sentence_b = mem_quote
            sentence_a = non_mem_quote
            label = 1

        row = {'actual_mem_quote': memorable_quote,
               'sentence_a': sentence_a,
               'sentence_b': sentence_b,
               'label': label,
               'confidence': 1.0}

        if include_metadata:
            row['title'] = title
            row['mem_line_no'] = int(memorable_line.split()[0].strip())
            row['nonmem_line_no'] = int(non_memorable_line.split()[0].strip())

        memorable_pairs.append(row)
    return memorable_pairs


def get_custom_eval(path_to_data: str, include_metadata: bool = False, exclude_equal_arguments=None):
    '''
    Extracts a custom dataset from a text file. Each line is a unique sentence, and all sentences are compared against each other.

    Args:
        directory (str): directory to the custom dataset
        include_metadata (bool): unnecessary
        exclude_equal_arguments (bool): unnecessary
    Returns:
        a list of dictionaries with both sentences and the label
    '''
    pairs = []
    with open(path_to_data) as f:
        lines = f.readlines()

    lines = list(filter(lambda a: a.strip() != '', lines))
    for sentence_a in lines:
        for sentence_b in lines:
            if sentence_a == sentence_b:
                continue

            row = {'sentence_a': sentence_a,
                   'sentence_b': sentence_b,
                   'label': -1,
                   'confidence': 0}  # all preds are marked incorrect
            pairs.append(row)

    return pairs


if __name__ == '__main__':

    persuasive_pairs = get_persuasive_pairs_xml(
        include_metadata=True, exclude_equal_arguments=True)
    filenames = sorted(set([filename.split('/')[-1][:10] for filename in glob.glob(
        './16k_persuasiveness/data/UKPConvArg1Strict-XML/' + '*.xml')]))
    for filename in filenames:
        labels = [row['label'] for row in persuasive_pairs if filename ==
                  row['filename'].split('/')[-1][:10]]
        # gets the number of labels for each file and aggregates them based on argument (both yes and no files)
        counts = [{label: labels.count(label)} for label in set(labels)]
        print(filename.split('/')[-1][:10], counts)

    # print(get_memorable_pairs_imdb('cornell_movie_quotes_corpus\moviequotes.memorable_nonmemorable_pairs.txt'))
