import argparse
import torch
import logging
from collections import OrderedDict

from transformers import BertTokenizer
from style_classifier.models import StyleClassifierScore

logging.basicConfig()
logger = logging.getLogger('log')
logger.setLevel(logging.INFO)

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str,
                        default='./sims/cmv_sim.txt', help="Data file.")
    parser.add_argument("--model", type=str,
                        default='./imdb_model.pth', help="Backbone model name/path")
    parser.add_argument("--k", type=int, default=5, help="k nearest neighbors")
    parser.add_argument("--thd", type=float, default=-0.1, help="threshold for inference confidence")


    return parser.parse_args()

def selection_inference(args):
    print('Loading model...')
    model = StyleClassifierScore(n_classes=2)  
    sd = torch.load(args.model)
    new_sd = OrderedDict()

    for key, value in sd['state_dict'].items():
        new_key = key.replace('module.', '')
        new_sd[new_key] = value
    sd = new_sd
 
    model.load_state_dict(sd)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = model.cuda()

    print('Reading data...')
    with open(args.data, 'r') as f:
        data = f.readlines()
     
    print('Inference...')
    with open('./sels/imdb_' + args.data.split('/')[-1][:3] + '_out.txt', 'w') as f:
        assert len(data) % (args.k+1) == 0
        for i in range(0, len(data), args.k + 1):
            if not (i/args.k) % 1000:
                logger.info(f"{i/(args.k + 1)} / {len(data)/(args.k + 1)}")
                print(f"{i/(args.k + 1)} / {len(data)/(args.k + 1)}")

            orig = data[i]
            for j in range(args.k):
                new_sent = data[i+j+1] 

                inputs = tokenizer(
                    orig,
                    new_sent,
                    add_special_tokens=True,
                    max_length=512,
                    padding='max_length',
                    return_token_type_ids=True,
                    return_tensors='pt'
                )
                if inputs['input_ids'].shape[-1] > 512:
                    for key, item in inputs.items():
                        batch_size, tokens = item.shape
                        new_item = torch.zeros(batch_size, 512)
                        new_item = item[:, (tokens-512):]
                        inputs[key] = new_item
                        
                inputs = {key: item.to("cuda") for key, item in inputs.items()}

                outputs = model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
                outputs = outputs.detach().cpu().data
                if outputs > args.thd:
                    f.write(orig.replace('\n',' ') + '\n' + new_sent.replace('\n',' ') + '\n' + str(outputs.item()) + '\n')
                    # remove this if you are not using cmv
                    break
    print('Done!')

if __name__ == '__main__':

    args = get_args()
    selection_inference(args)

    
