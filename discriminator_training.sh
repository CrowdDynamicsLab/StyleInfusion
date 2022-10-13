export TRAINING_CONDA_ENV=""

conda activate $TRAINING_CONDA_ENV
python train_classifier.py --do_train --do_eval --model_save_name="./model.pth" --save_model --epochs=5 --train_batch_size=8 --lr=1e-5