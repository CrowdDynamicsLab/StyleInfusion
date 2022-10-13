# style_data_augmentation

### Installation
To run the code in this repo, just create an environment with the requirements listed in `requirements.txt` or build a conda environment from the `environment.yml` file.

### Directory Structure
    .
    ├── .gitignore                   
    ├── debug.ipynb                # Testing Jupyter Notebook
    ├── emb.py                     # Retrieves embeddings of all datasets
    ├── load_data.py               # Loads/cleans all the datasets used
    ├── README.md   
    ├── requirements.txt
    ├── sim.py                     # Calculates similarities between embeddings
    └── sel.py                     # Selects the persuasive sentences to augmentvi 

### Running the Code

The first step is to run `emb.py` to generate the embeddings of the datasets. In the main call, you can choose which datasets to run and the program will store the embeddings in a `.csv` file. Note that we use the Universal Sentence Embeddings from Google, but you can exchange this with any embeddings.

The next step is to run the FAISS similarity search using `sim.py`. In the main call, you have to choose which datasets to use and at the top of the file, there is a `k_mapping` parameter which you should set. If you want to be more selective with the sentences you augment, choose a lower `k`. This will generate similarity files with 1 candidate sentence and `k` similar sentences in the reference dataset (usually 16k Persuasive Pairs).

Lastly, run the `sel.py` file with the following arguments:
- `--data`: the similarity files
- `--model`: a persuasiveness classifier model for inference 
- `--k`: the k you set in the config earlier
This file will select the candidate sentences to ultimately include in the augmented dataset. This will output a file with the candidate sentence, a similar sentence, and the confidence score of the model.

### How the Similarity Search Works

From a dataset, we extract every sentence as a candidate sentence. For each candidate sentence, we find the `k` most similar sentences in a reference dataset (usually 16k Persuasive Pairs). In order to include a candidate sentence, it must be more persuasive than at least one of the `k` similar sentences. We determine this by setting a threshold on the confidence of the classifier when it compares the candidate sentence with one of the `k` similar sentences.
