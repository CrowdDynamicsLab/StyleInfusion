
## BlaBla Resources

git clone https://github.com/novoic/blabla.git
cd blabla
pip install .

#### To use CoreNLP Features

Run the ./setup_corenlp.sh file in the aforementioned github repository.

After installation, or if you already have CoreNLP installed, let BlaBla know where to find it using export CORENLP_HOME=/path/to/corenlp. Make sure you have JDK installed.

## NLTK Resources

>>> import nltk
>>> nltk.download('punkt')
>>> nltk.download('averaged_perceptron_tagger')
>>> nltk.download('cmudict')
>>> nltk.download('brown')

## Spellchecker

To install the library use:

`pip install pyspellchecker`

## Textatistic

To install the library use:

`pip install textatistic`

## Spacy

Download the spacy model with:

`python -m spacy download en_core_web_sm`

## Running

To run without the features, either call:

`python data_feature_utils.py` or `python get_data_features.py --trained_model_path="../models/model.pth"`

To run with features, but without inference results, call:

`python get_data_features.py --trained_model_path="../models/model.pth" --features_path="features.csv"`

To run with features and inference results, call:

`python get_data_features.py --inference_path="inference.csv" --features_path="features.csv"`
