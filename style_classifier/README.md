# Style Classifier

A persuasiveness classifier built on top of the [16k persuasive pairs dataset](https://github.com/UKPLab/acl2016-convincing-arguments).

## Architectures
There are currently two supported architectures:

1. Concatenates the two arguments, passes them through BERT to get the embedding, and uses a FC layer to output a prediction of the class. The output is a score between 0 and 1.

2. Similar to [SBERT](https://arxiv.org/pdf/1908.10084.pdf), passes each argument through BERT, pools the embedding, concatenates the two embeddings and the absolute difference, and classifies with a FC layer. The output is usually scores of shape [batch_size, 2], but there is also support for a score between 0 and 1 (not fully implemented).

## Usage
Ensure all dependencies are installed by running

`pip install -r requirements.txt`

Have the persuasive pairs dataset downloaded from the repo above.

See examples of how to run from the `scripts` directory (not included for anonymity). 

A basic example of running would be:

`python train_classifier.py --do_train --do_eval --model_save_name="./model_noequal_singlegpu.pth" --save_model --epochs=5 --train_batch_size=8 --lr=1e-5`

Running evaluation on the [IMDB memorability dataset (Cristian Danescu-Niculescu-Mizil et al., 2012)](https://www.cs.cornell.edu/~cristian/memorability.html) can be done with:

`python train_classifier.py --do_eval --dataset="imdb" --test_data_dir="./cornell_movie_quotes_corpus/moviequotes.memorable_nonmemorable_pairs.txt" --trained_model_path="./models/model_noequal_singlegpu.pth" --load_model --epochs=1 --show_incorrect`

You can see `train_classifier.py` for a full list of arguments and what they do.

### Running with your own dataset

We use a standard template for loading datasets (see `get_data_loaders`). To load your own data you must create a function that takes in at least three arguments:
- directory (str): directory or file of your dataset
- include_metadata (bool): whether to include extra data aside from bare minimum
- exclude_equal_arguments (bool): whether to include the label indicating the arguments are equal

This function must return a list of dictionaries with both arguments (`sentence_a`, `sentence_b`) and the label (`label`).

You can then import and pass this function into the `get_data_loaders` function.
