import os
import torch
import logging
import argparse
import pandas as pd
from datasets import load_metric
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DATA_BASE_DIR = './'
MODEL_BASE_DIR = ''
MODELS_TO_USE = [
    'gpt2'
]

SUPPORTED_METRICS = ['rouge', 'meteor', 'perplexity']

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=float, help='Number of tokens to take from each input', default=0.2)
    parser.add_argument('--length', type=int, help='Amount of new words added to input', default=100)
    parser.add_argument('--device', type=str, help='Device to inference on', default='cuda')
    parser.add_argument('--test_file', type=str, help='File with test sentences/prompts', default='test.txt')
    parser.add_argument('--logging_file', type=str, help='File to log to', default='./evaluation_log')
    parser.add_argument('--arch', type=str, help='Model architecture', default='gpt2')
    parser.add_argument('--model_name', type=str, help='Model to evaluate', default="")
    args = parser.parse_args()
    return args

def read_test_set(args):
    with open(os.path.join(DATA_BASE_DIR, args.test_file)) as f:
        lines = f.readlines()

    prompts = []
    new_prompts = []
    gold = []
    for line in lines:
        # prompt, _, response = line.split('\t')
        response = line.strip()
        prompt = ''
        n_tokens = int(args.n if args.n >= 1 or not args.n else len(line.split()) * args.n)
        new_prompt = prompt + ' ' +  ' '.join(response.split(' ')[:n_tokens])
        new_prompt = new_prompt.replace('.', '').replace('!','').replace('?','')

        prompts.append(prompt)
        new_prompts.append(new_prompt)
        gold.append(response)

    return prompts, gold, new_prompts

def generate_predictions(args, prompts, model, tokenizer):
    outputs = []
    for inp in prompts:
        inputs = tokenizer(inp, return_tensors='pt').to(args.device)
        output = model.generate(
            **inputs, 
            max_length=args.length, 
            num_beams=5, 
            early_stopping=True
        )

        output = tokenizer.decode(output[0], skip_special_tokens=True)
        outputs.append(output)
    return outputs

def get_metric(args, references, metric_name='rouge', predictions=None, model_id=None):
    assert metric_name in SUPPORTED_METRICS
    metric = load_metric(metric_name)

    if 'perplexity' == metric_name:
        assert model_id is not None
        return metric.compute(input_texts=references, model_id=model_id, device=args.device)
    elif 'rouge' == metric_name:
        assert predictions is not None
        rouge_scores = metric.compute(predictions=predictions, references=references)
        rouge_scores = {k: v.high for k, v in rouge_scores.items()}

        split_rouge_scores = {}
        for key, val in rouge_scores.items():
            split_rouge_scores[key + '_P'] = val.precision
            split_rouge_scores[key + '_R'] = val.recall
            split_rouge_scores[key + '_F'] = val.fmeasure
        return split_rouge_scores
    else:
        assert predictions is not None
        return metric.compute(predictions=predictions, references=references)

def postprocess_preds(predictions, prompt_blacklist, blacklist, strict=False):
    new_preds = []
    for pred in predictions:
        # remove prompts
        for prompt in prompt_blacklist:
            if pred.startswith(prompt):
                pred = pred[len(prompt):]

            if strict:
                pred = pred.replace(prompt,'')

        # remove undesired phrases (e.g. CNN/DM)
        for phrase in blacklist:
            pred = pred.replace(phrase, '')

        pred = pred.strip().replace('\n', ' ') + '\n'
        new_preds.append(pred)
    return new_preds

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(filename=os.path.join(DATA_BASE_DIR, args.logging_file),
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)

    logger = logging.getLogger('evaluation_logger')

    arg_prompts, gold, prompts = read_test_set(args)
    tokenizer = GPT2Tokenizer.from_pretrained(args.arch)

    if len(args.model_name):
        MODELS_TO_USE = [args.model_name]

    model_scores = []
    for model_name in MODELS_TO_USE:
        if not os.path.exists(os.path.join(DATA_BASE_DIR, 'generations')):
            os.mkdir(os.path.join(DATA_BASE_DIR, 'generations'))

        prediction_path = os.path.join(DATA_BASE_DIR, 'generations', model_name.replace('/', '-') + '.txt')
        if os.path.exists(prediction_path):
            print(f'{model_name} predictions already exist!')
            logger.info(f'{model_name} predictions already exist!')

            with open(prediction_path, 'r') as f:
                predictions = f.readlines()

        else:
            logger.info(model_name + '\n')
            print(model_name + '\n')
            try:
                # model = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device)
                print(os.path.join(MODEL_BASE_DIR, model_name))
                model = GPT2LMHeadModel.from_pretrained(os.path.join(MODEL_BASE_DIR, model_name)).to(args.device)
            except OSError:
                print("Can't load this model right now to generate sentences.")
                continue
            predictions = generate_predictions(args, prompts, model, tokenizer)
            predictions = postprocess_preds(predictions, prompt_blacklist=arg_prompts, blacklist=['CMV'])

            with open(prediction_path, 'w') as f:
                for pred in predictions:
                    f.write(pred)

            model = model.cpu()
            del model

        scores = {'model': model_name}
        for metric in SUPPORTED_METRICS:
            try:
                results = get_metric(args, gold, metric, predictions, os.path.join(MODEL_BASE_DIR, model_name))
            
                logger.info(results)
                print(results)
                scores.update(results)
            except OSError:
                print("Can't load this model right now to calculate perplexity.")
        model_scores.append(scores)
    
    pd.DataFrame(model_scores).to_csv('scores.csv', mode='a')
    print(model_scores)
            

        

        
        



