import re
import os
import csv
from xml.etree.ElementInclude import include
import glob
import pandas as pd
from pickle import dump, load
from itertools import dropwhile
from typing import List, Dict, Tuple

import spacy
import nltk
from nltk.corpus import cmudict, brown
from nltk.tokenize import word_tokenize

from spellchecker import SpellChecker
from textatistic import Textatistic
from blabla.document_processor import DocumentProcessor
from transformers import pipeline

from style_classifier.utils.load_data import get_persuasive_pairs_xml

def update_dict_list(orig, extra):
    for i, e in enumerate(extra):
        orig[i].update(e)
    return orig


def get_features_dicts(texts, mtcg_verbs_path: str = 'mtcg_verbs.csv', stanza_config_path='stanza_config/stanza_config.yaml'):
    sen_lens = [len(text[-1]) for text in texts]

    features_dicts = [{'id': i, 'text': text, 'length': len(text)} for (i, text) in texts]

    texts = [text for (i, text) in texts]
    tagged_texts = []
    for text in texts:
        tokenized = nltk.word_tokenize(text)
        pos_tagged = nltk.pos_tag(tokenized)
        tagged_texts.append(pos_tagged)

    verb_counts = count_verbs(tagged_texts)
    verb_tenses = get_verb_tense(tagged_texts)

    tab = csv.DictReader(open(mtcg_verbs_path, 'r', encoding='utf-8-sig'))
    mtcg_verbs = get_mtcg_verbs(tab, texts)

    update_dict_list(features_dicts, get_ratios(verb_tenses, verb_counts))
    update_dict_list(features_dicts, get_ratios(verb_tenses, sen_lens))
    update_dict_list(features_dicts, get_ratios(mtcg_verbs, verb_counts))
    update_dict_list(features_dicts, get_ratios(mtcg_verbs, sen_lens))

    punct_dicts = count_punctuation(texts)
    punct_counts = [sum(punct_dict.values()) for punct_dict in punct_dicts]
    update_dict_list(features_dicts, get_ratios(punct_dicts, punct_counts))

    nlp = spacy.load("en_core_web_sm")
    named_entity_dicts = ner(nlp, texts)
    named_entity_counts = [sum(named_entity_dict.values())
                           for named_entity_dict in named_entity_dicts]
    update_dict_list(features_dicts, get_ratios(
        named_entity_dicts, named_entity_counts))

    readability_scores = get_readability_scores(texts)
    update_dict_list(features_dicts, readability_scores)

    update_dict_list(features_dicts, get_blabla_features(
        texts, corenlp=False, stanza_config=stanza_config_path))  # corenlp doesn't seem to work

    spell = SpellChecker()
    mispelled_counts = spell_check_text(spell, texts)
    mispelled_ratios = [{'mispelled': count / total}
                        for count, total in zip(mispelled_counts, sen_lens)]
    update_dict_list(features_dicts, mispelled_ratios)

    # nltk.download('cmudict') ## download CMUdict for phoneme set
    phoneme_dictionary = nltk.corpus.cmudict.dict()
    # nltk.download('stopwords') ## download stopwords (the, a, of, ...)
    stopwords = nltk.corpus.stopwords.words("english")

    alliteration_scores = get_alliteration_level(texts, stopwords, phoneme_dictionary)
    update_dict_list(features_dicts, alliteration_scores)

    tagger = Tagger()
    is_passive_sents = is_passive(tagger, texts)
    update_dict_list(features_dicts, is_passive_sents)

    cmu_dict = cmudict.dict()
    syllable_counts = get_syllables_sent(texts, cmu_dict)
    update_dict_list(features_dicts, syllable_counts)

    wordlist = set(brown.words())
    jargon_counts = get_jargon_counts(texts, wordlist)
    update_dict_list(features_dicts, get_ratios(jargon_counts, sen_lens))
    
    sentiments = get_sentiment(texts)
    update_dict_list(features_dicts, sentiments)

    return features_dicts


# blabla feature labels that do not rely on Stanford CoreNLP
FEATURES = ['noun_rate', 'verb_rate', 'demonstrative_rate', 'adjective_rate', 'adposition_rate', 'adverb_rate', 'auxiliary_rate',
            'conjunction_rate', 'determiner_rate', 'interjection_rate', 'numeral_rate', 'particle_rate', 'pronoun_rate',
            'proper_noun_rate', 'punctuation_rate', 'subordinating_conjunction_rate', 'symbol_rate', 'possessive_rate',
            'noun_verb_ratio', 'noun_ratio', 'pronoun_noun_ratio', 'closed_class_word_rate', 'open_class_word_rate',
            'total_dependency_distance', 'average_dependency_distance', 'total_dependencies', 'average_dependencies', 'content_density',
            'idea_density', 'honore_statistic', 'brunet_index', 'type_token_ratio', 'word_length', 'prop_inflected_verbs',
            'prop_auxiliary_verbs', 'prop_gerund_verbs', 'prop_participles']

# blabla feature labels that rely on Stanford CoreNLP
CORENLP_FEATURES = ['num_clauses', 'clause_rate', 'num_dependent_clauses',
                    'dependent_clause_rate', 'prop_nouns_with_det', 'prop_nouns_with_adj', 'num_noun_phrases', 'noun_phrase_rate',
                    'num_verb_phrases', 'verb_phrase_rate', 'num_infinitive_phrases', 'infinitive_phrase_rate', 'num_prepositional_phrases',
                    'prepositional_phrase_rate', 'max_yngve_depth', 'mean_yngve_depth', 'total_yngve_depth', 'parse_tree_height',
                    'num_discourse_markers', 'discourse_marker_rate']


def get_blabla_features(texts: List[str], corenlp: bool = False, stanza_config: str = 'stanza_config/stanza_config.yaml') -> List[Dict]:
    '''
    Passes a list of sentences through the blabla library's API to get a set of linguistic features.

    args:
        texts: list of sentences
        corenlp: a flag of whether to get features dependent on corenlp (doesn't work currently)
        stanza_config: directory to stanza's configuration file
    returns:
        a list of dictionaries with the keys being the feature names. each dictionary corresponds to a sentence
    '''
    if corenlp:
        raise NotImplementedError

    with DocumentProcessor(stanza_config, 'en') as doc_proc:
        doc = [doc_proc.analyze(line, 'string') for line in texts]

    res = [d.compute_features(
        FEATURES + (CORENLP_FEATURES if corenlp else [])) for d in doc]
    return res


def ner(nlp, texts: List[str]) -> List[Dict]:
    '''
    Performs named entity recognition on a list of texts

    args:
        nlp: spacy nlp processer
        texts: list of sentences
    returns:
        a list of dictionaries (each corresponds to a sentence) with the keys being entities and values being occurances.
    '''
    results = []
    for text in texts:
        doc = nlp(text)
        cnt = {}
        for ent in doc.ents:
            cnt[ent.label_] = cnt.get(ent.label_, 0) + 1

        results.append(cnt)
    return results


def count_verbs(tagged_texts: List[List[Tuple]]):
    '''
    Counts the number of verbs in a set of sentences

    args:
        tagged_texts: POS tagged sentences. Each tuple represents (word, tag)
    returns:
        a list of counts of verbs
    '''
    counts = []
    for text in tagged_texts:
        count = len([word for word in text if 'V' == word[1][0]])
        counts.append(count)
    return counts


def get_verb_tense(tagged_texts: List[List[Tuple]]) -> List[Dict]:
    '''
    Gets the verb tenses (past, present, future) for each verb in list of sentences
    TODO: consider spacy?

    args:
        tagged_texts: POS tagged sentences. Each tuple represents (word, tag)
    returns:
        a list of dictionaries containing the count of tenses for each sentence
    '''
    tense_counts = []
    for text in tagged_texts:
        tense = {}
        tense["future"] = len(
            [word for word in text if word[1] in ["VBC", "VBF", "MD"]])
        tense["present"] = len(
            [word for word in text if word[1] in ["VBP", "VBZ", "VBG"]])
        tense["past"] = len(
            [word for word in text if word[1] in ["VBD", "VBN"]])
        tense_counts.append(tense)

    return tense_counts


def get_mtcg_verbs(tab: Dict, texts: List[str]) -> List[Dict]:
    '''
    Gets the type of verb (modal, tentative, certainty, generalizing) for each verb in a set of sentences.

    args:
        tab: a dictionary mapping verb to their type of verb
        texts: list of sentences
    returns:
        a list of dictionaries containing the count of each type of verb for each sentence
    '''
    tabi = dict(sorted([(v.lower(), k) for e in tab for k, v in e.items()]))
    results = []

    for text in texts:
        verb_counts = {}
        for word in text.lower().split():
            if word in tabi:
                verb_counts[tabi[word]] = verb_counts.get(tabi[word], 0) + 1
        # print(', '.join([f"'{word}': ({cnt[word]}, {mod[word]})" for word, _ in sorted(cnt.items(), key = lambda e: e[0])]))
        results.append(verb_counts)
    return results


def get_ratios(counts, total_counts):
    for i, total in enumerate(total_counts):
        if total:
            counts[i] = {key: val/total for key, val in counts[i].items()}
        else:
            counts[i] = {key: 0 for key, _ in counts[i].items()}
    return counts


def count_punctuation(texts):
    results = []
    for text in texts:
        cnt = {}
        for punct in '.!?':
            cnt[punct] = cnt.get(punct, 0) + text.count(punct)
        results.append(cnt)
    return results


def spell_check_text(spell, texts):
    incorrect_counts = []
    for text in texts:
        incorrect_cnt = 0
        misspelled = spell.unknown(
            list(map(lambda x: re.sub("[^a-zA-Z]+", "", x).strip().lower(), text.split())))
        for word in misspelled:
            # Get the one `most likely` answer
            if word != spell.correction(word):
                incorrect_cnt += 1
                # print(word, spell.correction(word))
        incorrect_counts.append(incorrect_cnt)
    return incorrect_counts


def get_readability_scores(texts):
    scores = []
    for text in texts:
        text = text.strip() + '.'
        try:
            readability_scores = Textatistic(text).scores
            scores.append(readability_scores)
        except ZeroDivisionError:
            print(text)
            scores.append({})
        except ValueError:
            print(text)
            scores.append({})
    return scores


stress_symbols = ['0', '1', '2', '3...', '-', '!', '+', '/',
                      '#', ':', ':1', '.', ':2', '?', ':3']
# Get stopwords that will be discarded in comparison
# Function for removing all punctuation marks (. , ! * etc.)
no_punct = lambda x: re.sub(r'[^\w\s]', '', x)

def get_phonemes(word, phoneme_dictionary):
    if word in phoneme_dictionary:
        return phoneme_dictionary[word][0] # return first entry by convention
    else:
        return ["NONE"] # no entries found for input word

def get_alliteration_level(texts, stopwords, phoneme_dictionary): # alliteration based on sound, not only letter!
    alliteration_scores = []
    for text in texts:
        count, total_words = 0, 0
        proximity = 2 # max phonemes to compare to for consideration of alliteration
        i = 0 # index for placing phonemes into current_phonemes
        lines = text.split(sep="\n")
        for line in lines:
            current_phonemes = [None] * proximity
            for word in line.split(sep=" "):
                word = no_punct(word) # remove punctuation marks for correct identification
                total_words += 1
                if word not in stopwords:
                    if (get_phonemes(word, phoneme_dictionary)[0] in current_phonemes): # alliteration occurred
                        count += 1
                    current_phonemes[i] = get_phonemes(word, phoneme_dictionary)[0] # update new comparison phoneme
                    i = 0 if i == 1 else 1 # update storage index

        alliteration_score = count / total_words
        alliteration_scores.append({'alliteration_score': alliteration_score})
    return alliteration_scores

# Code from the `ispassive` library
# https://github.com/cowlicks/ispassive

class Tagger:
    def __init__(self):
        if os.path.exists("tagger.pkl"):
            with open('tagger.pkl', 'rb') as data:
                tagger = load(data)
            self.tagger = tagger
        else:
            tagger = create_tagger()
            self.tagger = tagger
            self.save()

    def save(self):
        with open('tagger.pkl', 'wb') as output:
            dump(self.tagger, output, -1)

    def tag(self, sent):
        return self.tagger.tag(sent)

    def tag_sentence(self, sent):
        """Take a sentence as a string and return a list of (word, tag) tuples."""
        tokens = nltk.word_tokenize(sent)
        return self.tag(tokens)

    def is_passive(self, sent):
        return is_passive(self, sent)

def passivep(tags):
    """Takes a list of tags, returns true if we think this is a passive
    sentence.
    Particularly, if we see a "BE" verb followed by some other, non-BE
    verb, except for a gerund, we deem the sentence to be passive.
    """
    
    after_to_be = list(dropwhile(lambda tag: not tag.startswith("BE"), tags))
    nongerund = lambda tag: tag.startswith("V") and not tag.startswith("VBG")

    filtered = filter(nongerund, after_to_be)
    out = any(filtered)

    return out

def create_tagger():
    """Train a tagger from the Brown Corpus. This should not be called very
    often; only in the event that the tagger pickle wasn't found."""
    train_sents = brown.tagged_sents()

    # These regexes were lifted from the NLTK book tagger chapter.
    t0 = nltk.RegexpTagger(
        [(r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
         (r'(The|the|A|a|An|an)$', 'AT'), # articles
         (r'.*able$', 'JJ'),              # adjectives
         (r'.*ness$', 'NN'),              # nouns formed from adjectives
         (r'.*ly$', 'RB'),                # adverbs
         (r'.*s$', 'NNS'),                # plural nouns
         (r'.*ing$', 'VBG'),              # gerunds
         (r'.*ed$', 'VBD'),               # past tense verbs
         (r'.*', 'NN')                    # nouns (default)
        ])
    t1 = nltk.UnigramTagger(train_sents, backoff=t0)
    t2 = nltk.BigramTagger(train_sents, backoff=t1)
    t3 = nltk.TrigramTagger(train_sents, backoff=t2)
    return t3

def is_passive(tagger, sents):
    passive_sents = []
    for sent in sents:
        tagged = tagger.tag_sentence(sent)
        tags = map(lambda tup: tup[1], tagged)
        sent_is_passive = int(bool(passivep(tags)))
        passive_sents.append({'is_passive': sent_is_passive})
    return passive_sents

def get_syllables_sent(sents, cmu_dict):
    sent_syllables = []
    for sent in sents:
        total_syl = 0
        for word in sent.split():
            total_syl += nsyl(word, cmu_dict)
        sent_syllables.append({'syllables': total_syl / len(sent.split())})
    return sent_syllables

def nsyl(word, cmu_dict):
    try:
        return [len(list(y for y in x if y[-1].isdigit())) for x in cmu_dict[word.lower()]][-1]
    except KeyError:
        #if word not found in cmudict
        return syllables(word)

def syllables(word):
    #referred from stackoverflow.com/questions/14541303/count-the-number-of-syllables-in-a-word
    count = 0
    vowels = 'aeiouy'
    word = word.lower()
    if word[0] in vowels:
        count +=1
    for index in range(1,len(word)):
        if word[index] in vowels and word[index-1] not in vowels:
            count +=1
    if word.endswith('e'):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count

def get_jargon_counts(sents, wordlist):
    jargon_counts = []
    for sent in sents:
        words = word_tokenize(sent)
        jargon_count = 0
        for word in words:
            if word not in wordlist:
                jargon_count += 1
        
        jargon_counts.append({'jargon': jargon_count})    
    return jargon_counts

def get_sentiment(sents):
    sentiments = []
    classifier = pipeline("sentiment-analysis")
    for sent in sents:
        res = classifier(sent)[0]
        sentiment = {'sentiment_label': 1 if res['label'] == 'POSITIVE' else 0, 'sentiment_score': res['score']}
        sentiments.append(sentiment)
    return sentiments
    

if __name__ == "__main__":

    for filename in glob.glob('./style-infusion/generations/*.txt'):
        print(filename)
        cleaned_filename = filename.replace('.txt', '').replace('./style-infusion/generations', 'features/imdb')
        skipping = False
        for feat_file in glob.glob('features/*.csv'):
            if cleaned_filename in feat_file:
                print(f'{filename} already has features!')
                skipping = True

        if skipping:
            continue

        
        with open(filename) as f:
            texts = f.readlines()
        
        texts = [(i, ''.join(text.split()[:256])) for i, text in enumerate(texts)]
        features_dict = get_features_dicts(texts)

        # get all potential keys from all the dictionaries in features_dict and put them in a list
        features_df = pd.DataFrame(features_dict)
        features_df.to_csv(cleaned_filename + '_features.csv')
