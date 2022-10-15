import torch
import torch.nn as nn
from torch.autograd import Variable

from models.batch_utils import decoded_batch_to_txt, get_output_from_batch, init_batch, run_decoder
from models.style_scorer import get_reward
from dutils.config import *


def get_rl_loss(args, batch, decoder, tokenizer, style_infusion_model, classifier_tokenizer, expected_reward_layer, tfidf_map, use_s_score, hidden_size=1024):
    print('Running RL loss...')
    device = torch.device('cuda', args['local_rank'])
    inputs, _, batch_size = init_batch(tokenizer, batch, device=device)

    step_mask = Variable(torch.ones(batch_size)).float()

    if USE_CUDA:
        step_mask = step_mask.to(device)

    all_step_mask = []
    all_targets = []
    all_output1 = []
    step_losses = []

    # in len of maximum size of headlines
    for di in range(args["max_r"]):
        inputs, outputs, final_dist, target, _ = run_decoder(
            decoder, tokenizer, inputs)
        # do this to avoid negatives being fed into multinomial
        final_dist = final_dist.softmax(dim=1)

        all_targets.append(target.detach())
        output1 = outputs['hidden_states'][-1][:, -1, :].to(device).float()
        # print(output1.shape)
        # this is some hidden state (batch * hidden_dim) -> o_t
        all_output1.append(output1)
        # gold_probs = final_dist[:, target]
        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + args["eps"])

        step_loss = step_loss * step_mask
        # print(f'gold_probs {gold_probs.data}\nstep_loss {step_loss.data}\nstep_mask {step_mask.data}\n')
        all_step_mask.append(step_mask)
        step_losses.append(step_loss)
        step_mask = torch.clamp(
            step_mask - (target == tokenizer.eos_token_id).float(), min=0.0)

    # this is the linear layer that calculates \hat{R}_t
    # TODO: logistic layer
    # sigmoid = nn.Sigmoid().to(device)
    baseline_rewards = [expected_reward_layer(output1.detach()) * step_mask.unsqueeze(1).detach()
                        for output1, step_mask in zip(all_output1, all_step_mask)]
    # batch size x decoding steps
    # for i, (output1, step_mask) in enumerate(zip(all_output1, all_step_mask)):
    #     print(f'Step {i}')
    #     print(f'Output {output1}\nERL {sigmoid(expected_reward_layer(output1.detach()))}\nStep {step_mask}')
    baseline_rewards = torch.cat(baseline_rewards, dim=1)
    all_step_mask = torch.stack(all_step_mask, dim=1).float()
    dec_lens_var = torch.sum(all_step_mask, dim=1)

    decoded_sents = decoded_batch_to_txt(tokenizer, all_targets)
    #print(f'Decoded: {decoded_sents}')
    total_reward = get_reward(
        decoded_sents, batch['target_txt'], style_infusion_model, classifier_tokenizer, device)
    # batch_size
    total_reward = total_reward.unsqueeze(1)

    # getting (R - \hat{R}_t)
    # torch.Size([batch, 1]) torch.Size([batch, max_r])
    # print(f'style_infusion {batch["style_infusion_scores"]}\ntotal_reward {total_reward}\nbaseline_reward {baseline_rewards}')
    # baseline_rewards = torch.nn.functional.softmax(baseline_rewards, dim=1)
    reward = torch.abs(total_reward.detach() - baseline_rewards.detach())
    # reward = total_reward.detach() - baseline_rewards.detach()
    # print(f'reward {reward}')
    sum_losses = torch.sum(reward * torch.stack(step_losses, 1), 1)
    # print(f'style_infusion {(1 - batch["style_infusion_scores"])}')
    # print(f'reward: {reward}\nsum: {torch.sum(torch.stack(step_losses, 1), 1)}\nsum_losses {sum_losses}\ndec_lens_var {dec_lens_var}')
    # this is for ARL
    if use_s_score:
        # use the model instead
        # batch_avg_loss = sum_losses / \
        #     dec_lens_var.float()*(1 - batch["style_infusion_scores"])
        batch_avg_loss = sum_losses / \
            dec_lens_var.float() * \
            ((1 - batch["style_infusion_scores"]) * args['beta'])
    else:
        batch_avg_loss = sum_losses/dec_lens_var.float()
    # print(batch_avg_loss)
    rl_loss = torch.mean(batch_avg_loss)
    ml_loss = get_loss(args, batch, decoder, tokenizer,
                       tfidf_map, use_s_score=use_s_score)
    print(f'rl_loss: {rl_loss}, ml_loss: {ml_loss}')
    if use_s_score:
        loss = rl_loss + ml_loss
    else:
        loss = (1 - args["ml_wt"]) * rl_loss + \
            args["ml_wt"] * ml_loss

    rewards_loss = torch.sum(
        (total_reward - baseline_rewards) ** 2 * all_step_mask) / torch.sum(all_step_mask)
    return total_reward.mean(), loss, rewards_loss


def get_supervised_loss(args, batch, decoder, tokenizer, style_infusion_model, classifier_tokenizer, tfidf_map, use_s_score):
    print('Running Supervised loss...')
    device = torch.device('cuda', args['local_rank'])
    inputs, targets, batch_size = init_batch(tokenizer, batch, device=device)
    target_batch, dec_padding_mask, _, dec_lens_var = get_output_from_batch(
        targets)

    step_losses = []
    all_targets = []

    step_mask = Variable(torch.ones(batch_size)).float()

    if USE_CUDA:
        step_mask = step_mask.to(device)

    # in len of maximum size of headlines
    for di in range(args["max_r"]):
        inputs, outputs, final_dist, target, _ = run_decoder(
            decoder, tokenizer, inputs)
        all_targets.append(target.detach())
        # print(inputs['input_ids'], inputs['input_ids'].shape)
        decoded_sents = decoded_batch_to_txt(tokenizer, all_targets)

        # decoded_sents = list(map(lambda x: x.split(), tokenizer.batch_decode(inputs['input_ids'])))
        if di == 0:
            step_loss = torch.zeros(batch_size, device=device)
        else:
            step_loss = get_reward(
                decoded_sents, batch['target_txt'], style_infusion_model, classifier_tokenizer, device)
        step_loss = step_loss.unsqueeze(1)

        step_loss = step_loss * step_mask
        step_losses.append(step_loss)
        step_mask = torch.clamp(
            step_mask - (target == tokenizer.eos_token_id).float(), min=0.0)

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)

    # this is for ARL
    if use_s_score:
        # use the model instead
        batch_avg_loss = sum_losses / \
            dec_lens_var.float() * \
            ((1 - batch["style_infusion_scores"]) * args['beta'])
    else:
        batch_avg_loss = sum_losses/dec_lens_var.float()

    supervised_loss = torch.mean(batch_avg_loss)
    ml_loss = get_loss(args, batch, decoder, tokenizer,
                       tfidf_map, use_s_score=use_s_score)

    if use_s_score:
        print('Using AP loss...')
        loss = supervised_loss + ml_loss
    else:
        loss = (1 - args["ml_wt"]) * supervised_loss + \
            args["ml_wt"] * ml_loss

    return loss


def get_loss(args, batch, decoder, tokenizer, tfidf_map, use_s_score=False):
    print('Running MLE loss...')
    # calculates MLE loss
    # seems like target and dec batches are the same
    device = torch.device('cuda', args['local_rank'])
    inputs, targets, _ = init_batch(tokenizer, batch, device=device)
    target_batch, dec_padding_mask, _, dec_lens_var = get_output_from_batch(
        targets)

    step_losses = []
    generated = []

    for di in range(min(targets['input_ids'].shape[-1], args["max_r"])):
        inputs, _, _, _, step_loss = run_decoder(
            decoder, tokenizer, inputs, labels=inputs['input_ids'])

        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

        if args['use_rep']:
            generated.append(inputs['input_ids'][:, -1])

        # Teacher forcing
        inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

    if args['use_rep']:
        # stacks the generated tokens (column-wise)
        generated = torch.stack(generated, dim=1)

        rep_loss = 0
        for i in range(generated.shape[0]):
            keys, counts = generated[i, :].unique(return_counts=True)
            token_keys = tokenizer.decode(keys).split(' ')
            for i, token in enumerate(token_keys):
                if token in tfidf_map and counts[i].item() > 1:
                    rep_loss += (counts[i].item() - 1) * \
                        tfidf_map[token.lower()]

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    if use_s_score:
        batch_avg_loss = sum_losses / \
            dec_lens_var.float() * \
            (1 - ((1 - batch["style_infusion_scores"]) * args['beta']))
    else:
        batch_avg_loss = sum_losses/dec_lens_var.float()
    loss = torch.mean(batch_avg_loss)

    return loss


def get_prob(args, decoder, tokenizer, batch):
    device = torch.device('cuda', args['local_rank'])
    inputs, targets, _ = init_batch(tokenizer, batch, device=device)
    target_batch, dec_padding_mask, _, dec_lens_var = get_output_from_batch(
        targets)

    step_losses = []

    for di in range(min(targets['input_ids'].shape[-1], args["max_r"])):
        inputs, _, final_dist, _, _ = run_decoder(decoder, tokenizer, inputs)

        target = target_batch[:, di]

        gold_probs = torch.gather(
            final_dist, 1, target.unsqueeze(1)).squeeze()
        step_loss = -torch.log(gold_probs + args["eps"])
        step_mask = dec_padding_mask[:, di]
        step_loss = step_loss * step_mask
        step_losses.append(step_loss)

        # Teacher forcing
        inputs['input_ids'][:, -1] = targets['input_ids'][:, di]

    sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
    batch_avg_loss = sum_losses/dec_lens_var.float()
    loss = torch.mean(batch_avg_loss)

    return loss
