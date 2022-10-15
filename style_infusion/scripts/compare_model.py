import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='Initial text for GPT2 model', required=True)
parser.add_argument('--length', type=int, help='Amount of new words added to input', default=50)
parser.add_argument('--arch', type=str, help='Model architecture', default='gpt2')
parser.add_argument('--mle_model', type=str, help='mle model', default='./style_infusion/mle/16k_hold_mle_trainer/checkpoint-100000')
parser.add_argument('--our_model', type=str, help='Our Model', default="./style_infusion/16k_cmv_cnn_hold_arl-1.0_trainer/checkpoint-100000")
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained(args.arch)
baseline = GPT2LMHeadModel.from_pretrained(args.arch)
mle = GPT2LMHeadModel.from_pretrained(args.mle_model)
ours = GPT2LMHeadModel.from_pretrained(args.our_model)

input_list = []
if '.txt' not in args.input:
    input_list = [args.input]
else:
    with open(args.input, "r") as f:
        input_list = f.readlines()
        input_list = list(filter(lambda x: len(x) != 0, map(lambda x: x.strip(), input_list)))

for inp in input_list:
    inputs = tokenizer(inp, return_tensors='pt')

    baseline_output = baseline.generate(
        **inputs, 
        max_length=args.length, 
        num_beams=5, 
        early_stopping=True
    )

    mle_output = mle.generate(
        **inputs, 
        max_length=args.length, 
        num_beams=5, 
        early_stopping=True
    )

    our_output = ours.generate(
        **inputs, 
        max_length=args.length, 
        num_beams=5, 
        early_stopping=True
    )

    print(100 * '-')
    print(f"Prompt: {inp}")
    print("Pretrained: " + tokenizer.decode(baseline_output[0], skip_special_tokens=True))
    print("100k iters MLE CMV+CNN: " + tokenizer.decode(mle_output[0], skip_special_tokens=True))
    print("100k iters ARL Beta(0.5): " + tokenizer.decode(our_output[0], skip_special_tokens=True))