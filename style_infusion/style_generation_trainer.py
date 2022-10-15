import torch
import transformers
from transformers import Trainer, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config, BertTokenizer, TrainingArguments, get_scheduler, AutoTokenizer, AutoModelWithLMHead, AutoConfig

import deepspeed
from transformers.deepspeed import HfDeepSpeedConfig

from dutils.config import USE_CUDA, get_args
from models.losses import get_rl_loss, get_loss, get_supervised_loss
from models.models import StyleClassifier, StyleClassifierScore
from dutils.data_utils import prepare_data_seq

import os
from numpy import random
random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)


class CustomTrainer(Trainer):
    def __init__(self, args, model, tokenizer, optimizers, train_dataloader, eval_dataloader, custom_args, style_infusion_model, classifier_tokenizer, tfidf_map):
        super(CustomTrainer, self).__init__(args=args, model=model, tokenizer=tokenizer, optimizers=optimizers)
        self.custom_args = custom_args
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.style_infusion_model = style_infusion_model

        if self.custom_args['use_rl']:
            self.expected_reward_layer = torch.nn.Linear(
                custom_args["hidden_size"], 1)
            self.rl_optimizer = torch.optim.Adam(self.expected_reward_layer.parameters(), lr=custom_args["rl_lr"])

            if USE_CUDA:
                self.expected_reward_layer.to(torch.device('cuda', custom_args['local_rank']))
            self.expected_rewards_loss = 0

        self.classifier_tokenizer = classifier_tokenizer
        self.tfidf_map = tfidf_map
        

    def get_train_dataloader(self):
        return self.train_dataloader

    def get_eval_dataloader(self, eval_dataset= None):
        return self.eval_dataloader

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.autocast_smart_context_manager():
            # loss = self.compute_loss(model, inputs)
            if self.custom_args['ml_wt'] == 1.0:
                loss = get_loss(self.custom_args, inputs, model, self.tokenizer, self.tfidf_map, use_s_score=self.custom_args["use_s_score"])
            elif not self.custom_args['use_rl']:
                loss = get_supervised_loss(self.custom_args, inputs, model, self.tokenizer, self.style_infusion_model, self.classifier_tokenizer, self.tfidf_map, use_s_score=self.custom_args["use_s_score"])
            else:
                _, loss, expected_reward_loss = get_rl_loss(self.custom_args, inputs, model, self.tokenizer, self.style_infusion_model,
                                                        self.classifier_tokenizer, self.expected_reward_layer, self.tfidf_map, use_s_score=self.custom_args["use_s_score"])

        if self.args.n_gpu > 1:
            loss = loss.mean()
            if self.custom_args['ml_wt'] != 1.0 and self.custom_args['use_rl']:
                expected_reward_loss = expected_reward_loss.mean()

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps
            if self.custom_args['ml_wt'] != 1.0 and self.custom_args['use_rl']:
                expected_reward_loss = expected_reward_loss / self.args.gradient_accumulation_steps

        if self.custom_args['ml_wt'] != 1.0 and self.custom_args['use_rl']:
            self.rl_optimizer.zero_grad()

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
            self.scaler.scale(expected_reward_loss).backward()
        elif self.use_apex:
            from apex import amp
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()

            with amp.scale_loss(expected_reward_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
            if self.custom_args['ml_wt'] != 1.0 and self.custom_args['use_rl']:
                expected_reward_loss = self.deepspeed.backward(
                    expected_reward_loss)
        else:
            loss.backward()
            if self.custom_args['ml_wt'] != 1.0 and self.custom_args['use_rl']:
                expected_reward_loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), self.custom_args["max_grad_norm"])
        if self.custom_args['ml_wt'] != 1.0 and self.custom_args['use_rl']:
            expected_reward_loss = expected_reward_loss.detach()
            self.rl_optimizer.step()
            self.expected_rewards_loss += expected_reward_loss.data
            print(loss, expected_reward_loss)
        else:
            print(loss)
        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.custom_args['ml_wt'] == 1.0:
            loss = get_loss(self.custom_args, inputs, model, self.tokenizer, use_s_score=self.custom_args["use_s_score"])
        elif not self.custom_args['use_rl']:
            loss = get_supervised_loss(self.custom_args, inputs, model, self.tokenizer, self.style_infusion_model, self.classifier_tokenizer, use_s_score=self.custom_args["use_s_score"])
        else:
            _, loss, _ = get_rl_loss(self.custom_args, inputs, model, self.tokenizer, self.style_infusion_model,
                                     self.classifier_tokenizer, self.expected_reward_layer, use_s_score=self.custom_args["use_s_score"])

        outputs = None
        if return_outputs:
            outputs = model(**inputs)

        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    custom_args = get_args()
    custom_args['generator'] = 'gpt2'
    custom_args['hidden_size'] = 1024
    custom_args['persuasivness_clasifier_path'] = "./imdb_model.pth"

    os.environ['MASTER_PORT'] = '12360'

    style_infusion_model = StyleClassifierScore()
    # sd = torch.load(custom_args['persuasivness_clasifier_path'])['state_dict']
    # state_dict = {key.replace('module.','') : value for key, value in sd.items()}
    # style_infusion_model.load_state_dict(state_dict)
    style_infusion_model = torch.load(custom_args['persuasivness_clasifier_path'])


    training_args = TrainingArguments(custom_args['save_path'], 
                                      per_device_train_batch_size=1,
                                      save_steps=10000,
                                    #   gradient_accumulation_steps=8,
                                      num_train_epochs=10,
                                      max_steps=custom_args['total_steps'],
                                      deepspeed=custom_args['ds_config'],
                                      fp16=True
                                    )
    # print(training_args)
    device = torch.device('cuda', custom_args['local_rank'])

    print('Loading data...')
    train_dataloader, eval_dataloader, max_r, tfidf_map = prepare_data_seq(custom_args['training_data'], custom_args['eval_data'], custom_args['batch_size'], thd=custom_args['thd'])
    print(f'Dataloader len {len(train_dataloader)}, max_r {max_r}')
    # custom_args['max_q'] = max_q
    # setting max decoding length
    custom_args['max_r'] = 40

    print('Loading gpt model...')
    config = AutoConfig.from_pretrained(
        custom_args['generator'], output_hidden_states=True, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(custom_args['generator'])
    bert_tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased")
    tokenizer.pad_token = tokenizer.eos_token
    config.pad_token_id = config.eos_token_id
    decoder = AutoModelWithLMHead.from_pretrained(custom_args['generator'], config=config)
    
    optimizer = torch.optim.Adam(params=decoder.parameters(), lr=training_args.learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10000
    )

    if USE_CUDA:
        style_infusion_model = style_infusion_model.to(device)
        decoder = decoder.to(device)
    
    
    dschf = HfDeepSpeedConfig(custom_args['ds_config'])
    engine, optimizer, training_dataloader, lr_scheduler = deepspeed.initialize(model=decoder, config_params=custom_args['ds_config'], optimizer=optimizer)
    # torch.distributed.init_process_group(backend='nccl')
    # decoder = DataParallelModel(decoder, device_ids=[0,1])
    # decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[custom_args['local_rank']], output_device=custom_args['local_rank'])
    # style_infusion_model = torch.nn.parallel.DistributedDataParallel(style_infusion_model, device_ids=[custom_args['local_rank']], output_device=custom_args['local_rank'])

    trainer = CustomTrainer(
        args=training_args,
        model=decoder,
        tokenizer=tokenizer,
        optimizers=(optimizer, lr_scheduler),
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        custom_args=custom_args,
        style_infusion_model=style_infusion_model,
        classifier_tokenizer=bert_tokenizer,
        tfidf_map=tfidf_map,
    )
    torch.cuda.empty_cache()
    transformers.logging.set_verbosity_info()
    checkpoint_path = os.path.join(custom_args['save_path'], custom_args['checkpoint_name'])
    if os.path.exists(checkpoint_path):
        train_result = trainer.train(checkpoint_path)
    else:
        train_result = trainer.train()
    trainer.save_model()
