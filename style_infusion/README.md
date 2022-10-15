# Style Infused Generator

This repository contains the code for the training of the language model that is to be infused with a style. 

## Usage

We recommend having at least two 32 GB GPUs to use for training. Ensure all dependencies are installed by running

`pip install -r requirements.txt`

### Dataset
Ensure you have the training dataset ready. Your data should be in a form such that every line is (tab-delimited):

`prompt style_score styled_generation`

For example:

`Firefox vs. Internet Explorer   0.645235       Firefox of course.`

The training data should be specified using the `--training_data` argument

### Running Training

You should run training with the following command:

`deepspeed --num_gpus=<NUM_GPUS> style_generation_trainer.py --batch_size <BATCH> --beta <BETA> --training_data <DATASET> --save_path <SAVE_PATH> --total_steps <TOTAL_STEPS>`

If you do not want sample dependent training, use the `--use_s_score 0` argument. You will need to specify the `--ml_wt` argument in this case! You should use the `--use_rl` parameter if you want to use the discriminator loss. Otherwise, you can exclude the argument to use the supervised loss.

We generally recommend training for at least a 100,000 steps but you can train for 10,000 and still see interesting generations.

A sample run may look like this:

`deepspeed --num_gpus=2 style_generation_trainer.py --batch_size 2 -eps 1e-5 --beta 0.8 --training_data dataset/16k_cnn_hold-firefox-marriage.txt --save_path ./style-infusion/16k_cnn_hold_sd-0.8_trainer --use_rl`

This run trains a GPT2 model with the sample-dependent discriminator loss (beta = 0.8) on the UKPConv1Arg + CNN/DM corpora (holding a few topics for testing).

### Running Evaluation

To run evaluation, use the following command:

`python dutils/evaluation.py --model_name <MODEL_NAME> --test_file <TEST_DATA>`

You may need to set some hyperparameters in the file to specify the location of your data and your model.

### Other Arguments

You can refer to `dutils/config.py` to see the purpose of all arguments and how to use them.