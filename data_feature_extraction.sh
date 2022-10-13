export DATA_FEATURE_CONDA_ENV=""

conda activate $DATA_FEATURE_CONDA_ENV
python data_feature_utils.py
python feature_differences.py