import argparse
import os
import fire

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset, SequentialSampler)
import torch.nn.functional as F
from tqdm import tqdm, trange

from transformers import (RobertaConfig, RobertaTokenizer, RobertaForSequenceClassification)
from transformers import set_seed

from datasets import load_dataset

SAVE_PATH = 'output/roberta/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def get_dataloader(tokenizer_args, tokenizer, padding, max_length, batch_size, truncation=True, labels=None, shuffle=True):
    features = tokenizer(*tokenizer_args, padding=padding, max_length=max_length, truncation=truncation)
    all_input_ids = torch.tensor([f for f in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features.attention_mask], dtype=torch.long)

    if labels is not None:
        all_labels = torch.tensor([f for f in labels], dtype=torch.long)
        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels) 
    else:
        tensor_dataset = TensorDataset(all_input_ids, all_attention_mask) 

    if shuffle:
        sampler = RandomSampler(tensor_dataset)
    else:
        sampler = SequentialSampler(tensor_dataset)

    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size)
    return dataloader

def process_hf_dataset(dataset, split, task_name, tokenizer, padding, max_length, batch_size, truncation=True, n=None, shuffle=True):
    eval_split_keys = {
                        'imdb': 'test',
                        'sst2': 'validation',
                        'yelp_polarity': 'test',
                        'mnli': 'validation_matched',
                        'hans': 'validation',
                        'snli': 'validation',
                        'rte': 'validation'
            }
    if split == 'eval':
        split = eval_split_keys[task_name]

    tasks_to_keys = {
                    'imdb': ('text', None),
                    'yelp_polarity': ('text', None),
                    'sst2': ('sentence', None),
                    'mnli': ('premise', 'hypothesis'),
                    'hans': ('premise', 'hypothesis'),
                    'snli': ('premise', 'hypothesis'),
                    'rte': ('sentence1', 'sentence2')
            }

    sentence1_key, sentence2_key = tasks_to_keys[task_name]
    args = ((dataset[split][sentence1_key][:n],) if sentence2_key is None else (dataset[split][sentence1_key][:n], dataset[split][sentence2_key][:n]))
    labels = dataset[split]['label'][:n]
    return get_dataloader(args, tokenizer, padding, max_length, batch_size, truncation=truncation, labels=labels, shuffle=shuffle)

def process_custom_dataset(dataset, task_name, tokenizer, padding, max_length, batch_size, truncation=True, n=None, shuffle=True):
    tasks_to_keys = {
                        'counterfactual-imdb': ('Text', None, 'Sentiment'),
                        'mnli': ('sentence1', 'sentence2', 'label')
            }

    sentence1_key, sentence2_key, labels_key = tasks_to_keys[task_name]

    dataset[sentence1_key] = dataset[sentence1_key].astype(str)
    if sentence2_key is not None:
        dataset[sentence2_key] = dataset[sentence2_key].astype(str)

    args = ((dataset[sentence1_key].tolist()[:n],) if sentence2_key is None else (dataset[sentence1_key].tolist()[:n], dataset[sentence2_key].tolist()[:n]))

    features = tokenizer(*args, padding=padding, max_length=max_length, truncation=truncation)
    if task_name != 'mnli':
        labels = pd.Categorical(dataset[labels_key], ordered=True).codes.tolist()[:n]
    else:
        labels = dataset[labels_key].tolist()[:n]

    all_input_ids = torch.tensor([f for f in features.input_ids], dtype=torch.long)
    all_attention_mask = torch.tensor([f for f in features.attention_mask], dtype=torch.long)
    all_labels = torch.tensor([f for f in labels], dtype=torch.long)

    tensor_dataset = TensorDataset(all_input_ids, all_attention_mask, all_labels)
    if shuffle:
        sampler = RandomSampler(tensor_dataset)
    else:
        sampler = SequentialSampler(tensor_dataset)
    dataloader = DataLoader(tensor_dataset, sampler=sampler, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def process_lm_dataset(dataset_path, tokenizer, padding, max_length, batch_size, truncation=True, num_label_chars=1, n=None, shuffle=True):
    # label in first column, and text in rest of the columns
    dataset_texts, labels = [], []
    with open(dataset_path) as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if num_label_chars > 0:
                # print(f'{idx}: {line[:num_label_chars]}')
                try:
                    labels.append(int(line[:num_label_chars]))
                except Exception as e:
                    print(e)
                    print(idx, line)
            dataset_texts.append(line[num_label_chars:].replace(' <|endoftext|>', '').lstrip())
    dataset_texts, labels = dataset_texts[:n], labels[:n]
    args = ((dataset_texts,))
    if len(labels) == 0:
        labels = None
    return get_dataloader(args, tokenizer, padding, max_length, batch_size, truncation=truncation, labels=labels, shuffle=shuffle)

def train(model, tokenizer, optimizer, criterion, device, train_loader, num_epochs, output_dir):
    losses = []
    train_iterator = trange(int(num_epochs), desc='Epoch')
    for _ in train_iterator:
        tr_loss = 0
        step = None
        epoch_iterator = tqdm(train_loader, desc='Iteration')
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device), 'labels': batch[2].to(device)}
            labels = batch[2].to(device)

            optimizer.zero_grad()

            out = model(**inputs)[1].double().to(device)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()
        losses.append(tr_loss/(step+1))
        print('train loss: {}'.format(tr_loss/(step+1)))

    # save model and tokenizer
    print('Saving model and tokenizer')

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def eval(model, eval_loader, device, criterion=nn.CrossEntropyLoss(), with_labels=True):
    probs = None
    gold_labels = None

    eval_loss = 0
    step = None
    eval_iterator = tqdm(eval_loader, desc='Evaluating')
    for step, batch in enumerate(eval_iterator):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0].to(device), 'attention_mask': batch[1].to(device)}
            # out = model(**inputs)[0].double().to(device)

            out = model(**inputs)[0].double()
            out = F.softmax(out, dim=1)

            if with_labels:
                # inputs['labels'] = batch[2].to(device)
                labels = batch[2].to(device)
                loss = criterion(out, labels)

            if probs is None:
                probs = out.detach().cpu().numpy()
                if with_labels:
                    gold_labels = labels.detach().cpu().numpy()
            else:
                probs = np.append(probs, out.detach().cpu().numpy(), axis=0)
                if with_labels:
                    gold_labels = np.append(gold_labels, labels.detach().cpu().numpy(), axis=0)

            if with_labels:
                eval_loss += loss.item()
    
    if with_labels:
        eval_loss /= (step+1)
        print('eval loss: {}'.format(eval_loss))

        # compute accuracy
        preds = np.argmax(probs, axis=1)
        accuracy = np.sum(preds == gold_labels)/len(preds)
        print('eval accuracy: {}'.format(accuracy))

    return probs

def main():
    # create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_name', help='Task to fine-tune RoBERTa on', default='sst2')
    parser.add_argument('--roberta_version', type=str, default='roberta-base', help='Version of RoBERTa to use')
    parser.add_argument('--cache_dir_data', type=str, default='cache/huggingface/datasets', help='Path to cache directory')
    parser.add_argument('--cache_dir', type=str, default='cache/huggingface/transformers', help='Path to cache directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to fine-tune')
    parser.add_argument('--max_seq_length', type=int, default=None, help='Maximum sequence length of the inputs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Adam learning rate')
    parser.add_argument('--output_dir', type=str, default='roberta_ckpts/', help='Directory to save fine-tuned models')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for initialization')
    parser.add_argument('--file_format', type=str, default='.tsv', help='Data file format for tasks not available for download at HuggingFace Datasets')
    parser.add_argument('--train_file', type=str, default=None, help='LM txt file')
    parser.add_argument('--val_file', type=str, default=None, help='LM txt file')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of labels in training data')
    parser.add_argument('--fname', type=str, default=None, help='MSP output file')
    parser.add_argument('--n', type=int, default=None, help='Number of examples to process (for debugging)')

    args = parser.parse_args()

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available():
        args.cache_dir = args.cache_dir_data = None

    # huggingface and glue datasets
    hf_datasets = ['imdb', 'sst2', 'mnli', 'hans', 'snli', 'rte']
    glue = ['sst2', 'mnli', 'rte']

    # custom dataset label keys
    label_keys = {
                    'counterfactual-imdb': 'Sentiment',
            }

    # load dataset
    print('Loading dataset')

    if args.train_file is None:
        if args.task_name in hf_datasets:
            dataset = load_dataset(args.task_name, cache_dir=args.cache_dir_data) if args.task_name not in glue else load_dataset('glue', args.task_name, cache_dir=args.cache_dir_data)
            num_labels = dataset['train'].features['label'].num_classes
            print("num_labels =", num_labels)
        elif args.file_format == '.tsv':
            train_df = pd.read_table(os.path.join(os.getcwd(), args.task_name + '_train' + args.file_format))
            eval_df = pd.read_table(os.path.join(os.getcwd(), args.task_name + '_val' + args.file_format))
            num_labels = len(np.unique(pd.Categorical(train_df[label_keys.get(args.task_name, 'label')], ordered=True)))
    else:
        num_labels = args.num_labels

    # set seed
    set_seed(args.seed)

    # load RoBERTa tokenizer and model
    print('Loading RoBERTa tokenizer and model')

    config = RobertaConfig.from_pretrained(args.roberta_version, num_labels=num_labels, cache_dir=args.cache_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.roberta_version, cache_dir=args.cache_dir)
    model = RobertaForSequenceClassification.from_pretrained(args.roberta_version, config=config, cache_dir=args.cache_dir).to(device)

    # process dataset
    print('Processing dataset')

    padding = 'max_length'
    with_labels = True
    if args.train_file is None:
        if args.task_name in hf_datasets:
            train_loader = process_hf_dataset(dataset, 'train', args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size, n=args.n)
            # train_loader = None
            eval_loader = process_hf_dataset(dataset, 'eval', args.task_name, tokenizer, padding, args.max_seq_length, args.batch_size, n=args.n)
        elif args.file_format == '.tsv':
            task_name = args.task_name
            if 'mnli' in args.task_name:
                task_name = 'mnli'
            train_loader = process_custom_dataset(train_df, task_name, tokenizer, padding, args.max_seq_length, args.batch_size, n=args.n)
            eval_loader = process_custom_dataset(eval_df, task_name, tokenizer, padding, args.max_seq_length, args.batch_size, n=args.n)
    else:
        train_loader = process_lm_dataset(args.train_file, tokenizer, padding, args.max_seq_length, args.batch_size, n=args.n)
        if args.val_file is not None:
            eval_loader = process_lm_dataset(args.val_file, tokenizer, padding, args.max_seq_length, args.batch_size, n=args.n, num_label_chars=0)
        else:
            eval_loader = None
        with_labels = False

    # instantiate optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # fine-tune model 
    if train_loader is not None:
        print('Fine-tuning model')
        train(model, tokenizer, optimizer, criterion, device, train_loader, args.num_epochs, args.output_dir)

    # evaluate model
    if eval_loader is not None:
        print('Evaluating model')
        probs = eval(model, eval_loader, device, criterion, with_labels=with_labels)
        np.save(os.path.join(SAVE_PATH, f'{args.fname}_probs'), probs)
        msp = np.max(probs, axis=1)
        if args.fname is not None:
            np.save(os.path.join(SAVE_PATH, f'{args.fname}_msp'), msp)


if __name__ == '__main__':
    main()
    print("\n\n--------DONE--------")
