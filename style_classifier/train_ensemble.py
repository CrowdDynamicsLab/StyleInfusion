import argparse
import torch
import random
import logging

import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from scipy import stats
from transformers import BertTokenizer, get_scheduler

from utils.load_data import get_persuasive_pairs_xml, get_memorable_pairs_imdb, get_custom_eval, PersuasivePairsDataset, SiamesePersuasivePairsDataset
from models.models import StyleClassifier, SiameseStyleClassifier

from train_classifier import calculate_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Training on: ', device)

def limit_dataset(dataset, limit=20):
    '''
    Limits the number of samples for a dataset based on labels

    Args:
        dataset: list of dictionaries containing data
        limit: integer representing limit for each class
    '''
    new_dataset = []
    random.shuffle(dataset)
    labels = set([row['label'] for row in dataset])
    for label in labels:
        label_set = [row for row in dataset if row['label'] == label][:limit]
        new_dataset.extend(label_set)
    return new_dataset


def get_data_loaders(args, tokenizer, dataset_type=PersuasivePairsDataset,
                     data_loader_fn=get_persuasive_pairs_xml,
                     train_dir='./16k_persuasiveness/data/UKPConvArg1Strict-XML/',
                     test_dir='./16k_persuasiveness/data/UKPConvArg1Strict-XML/', 
                     limit=20):
    '''
    Gets the train and test dataloaders from the data loader function.

    Args:
        args: command-line arguments
        tokenizer: tokenizer for tokenizing samples
        dataset_type: type of dataset corresponding to model type
        data_loader_fn: function to load data from directories
        train_dir: directory to load training data from
        test_dir: directory to load testing data from
        limit: integer for limit on each class
    Returns:
        a train and test dataloader
    '''

    train_dataset, test_dataset = None, None
    train_dataloader, test_dataloader = None, None

    if train_dir == test_dir:
        # data is in the same file/directory
        persuasive_pairs = data_loader_fn(
            train_dir,
            exclude_equal_arguments=args.exclude_equal_arguments
        )

        # random train/test split
        if args.do_train and args.do_eval:
            total_length = len(persuasive_pairs)
            random.shuffle(persuasive_pairs)
            split = int(args.train_test_split * total_length)
            train_dataset = persuasive_pairs[:split]
            test_dataset = persuasive_pairs[split:]
        elif args.do_train:
            train_dataset = persuasive_pairs
        elif args.do_eval:
            test_dataset = persuasive_pairs
    else:
        if args.do_train:
            train_dataset = data_loader_fn(
                train_dir,
                exclude_equal_arguments=args.exclude_equal_arguments
            )
        if args.do_eval:
            test_dataset = data_loader_fn(
                test_dir,
                exclude_equal_arguments=args.exclude_equal_arguments
            )
    if train_dataset:
        train_dataset = limit_dataset(train_dataset, limit)
    if dataset_type == PersuasivePairsDataset:
        persuasive_pairs_train = dataset_type(train_dataset, tokenizer)
        persuasive_pairs_test = dataset_type(test_dataset, tokenizer)
    else:
        persuasive_pairs_train = dataset_type(train_dataset)
        persuasive_pairs_test = dataset_type(test_dataset)

    if args.do_train:
        train_dataloader = DataLoader(persuasive_pairs_train, batch_size=args.train_batch_size,
                                      shuffle=True, num_workers=0)
    if args.do_eval:
        test_dataloader = DataLoader(persuasive_pairs_test, batch_size=args.test_batch_size,
                                     shuffle=False, num_workers=0)

    return train_dataloader, test_dataloader


def fine_tune(args):
    # load in data
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset_type = SiamesePersuasivePairsDataset if args.architecture.lower(
    ) == "siamese" else PersuasivePairsDataset

    data_loader_fn = get_persuasive_pairs_xml
    if args.dataset == 'imdb':
        data_loader_fn = get_memorable_pairs_imdb 
    elif args.dataset == 'custom':
        data_loader_fn = get_custom_eval

    train_dataloaders = []
    models = []
    optimizers = []
    lr_schedulers = []
    for i in range(args.ensemble_size):
        # train and test dir need to be different
        args.do_eval = False
        train_dataloader, _ = get_data_loaders(
            args, tokenizer, dataset_type=dataset_type, data_loader_fn=data_loader_fn, train_dir=args.train_data_dir, limit=args.label_limit)
        train_dataloaders.append(train_dataloader)

        # create model
        if args.architecture.lower() == "siamese":
            model = SiameseStyleClassifier(args.model_name_or_path, tokenizer, n_classes=int(
                2 if args.exclude_equal_arguments else 3), device=device).to(device)
        else:
            model = StyleClassifier(args.model_name_or_path, dropout=args.dropout, n_classes=int(
                2 if args.exclude_equal_arguments else 3)).to(device)
        # load model
        model = torch.load(
            args.trained_model_path.replace('.pth', f'{i}.pth')) if args.trained_model_path and args.load_model else model


        # if there are multiple GPUs, distribute training
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(
                model, device_ids=list(range(torch.cuda.device_count())))

        # set up loss, optimizer, and lr scheduler

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        if train_dataloader is not None:
            num_training_steps = args.epochs * len(train_dataloader)
        else:
            num_training_steps = 0

        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        for param in model.parameters():
            param.requires_grad = True

        models.append(model)
        optimizers.append(optimizer)
        lr_schedulers.append(lr_scheduler)

    loss_fn = torch.nn.CrossEntropyLoss()

    args.do_eval = True
    args.do_train = False
    _, test_dataloader = get_data_loaders(
            args, tokenizer, dataset_type=dataset_type, data_loader_fn=data_loader_fn, test_dir=args.test_data_dir)
    args.do_train = True

    for epoch in range(args.epochs):
        if args.do_train:
            training(args, models, train_dataloaders, optimizers,
                     lr_schedulers, loss_fn, epoch)
        if args.do_eval:
            evaluation(args, models, test_dataloader, tokenizer,
                       show_incorrect=args.show_incorrect)

    if args.save_model:
        for i, model in enumerate(models):
            torch.save(model, args.model_save_name.replace('.pth', f'{i}.pth'))


def training(args, models, train_dataloaders, optimizers, lr_schedulers, loss_fn, epoch):
    for model, train_dataloader, optimizer, lr_scheduler in zip(models, train_dataloaders, optimizers, lr_schedulers):
        model.train()
        for _, data in enumerate(train_dataloader):
            if args.architecture.lower() == "siamese":
                texts = list(data['texts'])
                label = data['label'].to(device, dtype=torch.long)
                outputs = model(texts)
            else:
                ids = data['ids'].to(device, dtype=torch.long)
                mask = data['mask'].to(device, dtype=torch.long)
                token_type_ids = data['token_type_ids'].to(
                    device, dtype=torch.long)
                label = data['label'].to(device, dtype=torch.long)

                outputs = model(ids, mask, token_type_ids)

            optimizer.zero_grad()
            loss = loss_fn(outputs, label)

            if _ % 100 == 0:
                print(f'Epoch: {epoch}, Loss:  {loss.item()}')
                logging.info(f'Epoch: {epoch}, Loss:  {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        lr_scheduler.step()


def evaluation(args, models, test_dataloader, tokenizer, show_incorrect=False):

    test_labels = []
    test_outputs = []

    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            label = data['label'].to(device, dtype=torch.long)
            preds = []
            for model in models:
                model.eval()
                if args.architecture.lower() == "siamese":
                    texts = list(data['texts'])
                    
                    outputs = model(texts)
                else:
                    ids = data['ids'].to(device, dtype=torch.long)
                    mask = data['mask'].to(device, dtype=torch.long)
                    token_type_ids = data['token_type_ids'].to(
                        device, dtype=torch.long)

                    outputs = model(ids, mask, token_type_ids)
            # print(outputs.shape)
                pred = torch.argmax(outputs, dim=1).cpu().detach().numpy()
                preds.append(pred)
            preds = np.stack(preds, axis=0)
            preds = stats.mode(preds, 0)[0][0]

            gold_labels = label.cpu().detach().numpy()
            if show_incorrect:
                confidence = data['confidence']
                # print out the incorrect predictions
                correct_preds = list(gold_labels == preds)
                for i, pred in enumerate(correct_preds):
                    if not pred:
                        print(f'Model Confidence: {outputs[i]}')
                        if args.architecture.lower() == "siamese":
                            argument_a = texts[0][i]
                            argument_b = texts[1][i]
                        else:
                            arguments = tokenizer.decode(ids[i]).replace(
                                '[PAD] ', '').replace('[CLS] ', '')
                            argument_a, argument_b, * \
                                _ = arguments.split(' [SEP] ')
                        
                        print(
                            f'Argument 1: {argument_a}\nArgument 2: {argument_b}\nPrediction: {preds[i]}, Actual: {gold_labels[i]}, Confidence: {confidence[i]}\n')
            test_labels.extend(gold_labels)
            test_outputs.extend(preds)

    accuracy = calculate_accuracy(
        np.array(test_labels), np.array(test_outputs))
    confusion_mat = confusion_matrix(
        np.array(test_labels), np.array(test_outputs))
    print(confusion_mat)
    logging.info(str(confusion_mat))
    accuracy = 'Accuracy: ' + str(accuracy)
    print(accuracy)
    logging.info(accuracy)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--architecture", type=str,
                        default='concatenation', help="Architecture of the model (siamese or concatenation is supported).")
    parser.add_argument("--model_name_or_path", type=str,
                        default='bert-base-uncased', help="Backbone model name/path")
    parser.add_argument("--ensemble_size", type=int, default=5)

    parser.add_argument("--load_model", action="store_true",
                        help="Whether to load a model.")
    parser.add_argument("--trained_model_path", type=str, default='./model.pth',
                        help="Already trained persuasive classifier path. Ensure the --load_model flag is used.")

    parser.add_argument("--save_model", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--model_save_name", type=str,
                        default='./model.pth', help="Name of saved model. Ensure the --save_model flag is used.")

    parser.add_argument("--exclude_equal_arguments", action="store_true",
                        help="Whether to exclude the equal class (0).")
    parser.add_argument("--label_limit", type=int, default=20)
    parser.add_argument("--dataset", type=str,
                        default='persuasive',
                        help="Dataset to load from (supported: imdb or persuasive).")
    parser.add_argument("--train_data_dir", type=str,
                        default='./16k_persuasiveness/data/UKPConvArg1Strict-XML/',
                        help="Directory to load train data from.")
    parser.add_argument("--test_data_dir", type=str,
                        default='./16k_persuasiveness/data/UKPConvArg1Strict-XML/',
                        help="Directory to load test data from.")

    parser.add_argument("--train_test_split", type=float, default=0.8)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", default=0.2, type=float,
                        help="Dropout for fully-connected layers")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run eval on the test set.")
    parser.add_argument("--show_incorrect", action="store_true",
                        help="Whether to print incorrectly printed samples.")

    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument("--training_log_name", type=str,
                        default='logs/training.log', help="Name of training log.")

    args = parser.parse_args()
    logging.basicConfig(filename=args.training_log_name, level=logging.DEBUG)

    if args.no_cuda:
        device = torch.device('cpu')
    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if not args.no_cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    fine_tune(args)
