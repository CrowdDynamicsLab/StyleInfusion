export AUGMENTATION_CONDA_ENV=""

conda activate $AUGMENTATION_CONDA_ENV
python emb.py

python sim.py imdb
python sim.py 16k

python sel.py --data sims/imdb_cnn_sims.txt --model <MODEL_PATH> --k 3
python sel.py --data sims/16k_cnn_sims.txt --model <MODEL_PATH> --k 3

python create_dataset.py imdb sims/imdb_cnn_sims.txt
python create_dataset.py 16k sims/16k_cnn_sims.txt