import torch
import torch.nn as nn
from transformers import BertModel

from dutils.config import *

def get_reward(decoded_sents, target_sents, style_infusion_model, tokenizer, device=None):
    '''
    Gets R_sen
    '''

    if device is None:
        device = "cuda"

    joined_decoded_sents = [' '.join(sent) for sent in decoded_sents]
    sents = [pred + ' [SEP] ' + target for pred,
                target in zip(joined_decoded_sents, target_sents)]
    batch = tokenizer(
        sents, return_tensors='pt', padding=True)
    
    if batch['input_ids'].shape[-1] > 512:
        for key, item in batch.items():
            batch_size, tokens = item.shape
            new_item = torch.zeros(batch_size, 512)
            new_item = item[:, (tokens-512):]
            batch[key] = new_item


    if USE_CUDA:
        staging_device = next(style_infusion_model.parameters()).device
        # staging_device = style_infusion_model.device_ids[0]
        style_infusion_model = style_infusion_model.to(staging_device)
        batch = {key: item.to(staging_device) for key, item in batch.items()}

    try:
        rewards = style_infusion_model(batch)
    except RuntimeError:
        print('Runtime Error!')
        print(f'decoded: {decoded_sents}')
        print(f'decoded_lens: {[len(sent) for sent in decoded_sents]}')
        raise RuntimeError

    w = torch.FloatTensor([len(set(word_list)) * 1. / len(word_list)
                            if len(word_list) else 1 for word_list in decoded_sents])
    if USE_CUDA:
        rewards = rewards.to(device)
        w = w.to(device)
    style_infusion_reward = rewards * w
    return style_infusion_reward.detach()