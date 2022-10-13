export TRAINING_CONDA_ENV=""
export FEATURE_CONDA_ENV=""

export DATASET="<PATH_TO_DATASET>"

export HF_TRAINER_DIR="./save_dir/"

# For R, set SAMPLEDEP to 0 and ML_WT to 1.0
# For D, include the --use_rl flag
export BATCH=2
export TOTAL_STEPS=10000
export SAMPLEDEP=1
# if SAMPLEDEP, then set BETA
export BETA=0.1
# otherwise, set ML_WT
export ML_WT=1.0

if [[ $SAMPLEDEP -eq 0 ]]
then
    export MODEL_NAME="imdb_cnn_mle-${ML_WT}_trainer"
else
    export MODEL_NAME="imdb_cnn_sd-${BETA}_trainer"
fi

conda activate $TRAINING_CONDA_ENV
# SUPERVISED LOSS
# deepspeed --num_gpus=2 style_generation_trainer.py --batch_size $BATCH --use_s_score $SAMPLEDEP --ml_wt $ML_WT --beta $BETA --training_data $DATASET --save_path $HF_TRAINER_DIR/$MODEL_NAME --total_steps $TOTAL_STEPS

# DISCRIMINATOR LOSS
deepspeed --num_gpus=2 style_generation_trainer.py --batch_size $BATCH --use_s_score $SAMPLEDEP --use_rl --beta $BETA --training_data $DATASET --save_path $HF_TRAINER_DIR/$MODEL_NAME --total_steps $TOTAL_STEPS

python dutils/evaluation.py --model_name $MODEL_NAME/checkpoint-$TOTAL_STEPS

conda deactivate
conda activate $FEATURE_CONDA_ENV
cd ../style_classifier/data_feature_extraction
export CORENLP_HOME=""
python data_feature_utils.py
python feature_differences.py