import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import fire
import time
from roberta_fine_tune import eval, process_hf_dataset, process_lm_dataset, process_custom_dataset

SAVE_PATH = 'output/msp/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def process_entailment(dataset, tokenizer, key1='sentence1', key2='sentence2'):
    dataset_texts = []
    for ex in dataset:
        dataset_texts.append(ex[key1] + ' ' + ex[key2])
    return [encode(tokenizer, text) for text in dataset_texts]

def encode(tokenizer, text):
    return tokenizer.encode_plus(
      text,
      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
      return_token_type_ids=False,
      max_length=150,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',  # Return PyTorch tensors
    )

def process_msp(all_encodings, model):
    scores = []
    for encoding in tqdm(all_encodings):
        input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
        out = model(input_ids, attention_mask)[0]
        score = F.softmax(out[0], dim=0)
        scores.append(score.detach().cpu().numpy())
    max_probs = np.max(np.array(scores), axis=1)
    return max_probs

def main(model_path, val_file=None, dataset_name=None, dataset_config_name=None, split='eval', batch_size=16, max_length=None, n=None, fname='sample', cache_dir='/scratch/ua388/cache/huggingface/datasets', save_msp=True, alpha=None):
    if alpha is not None:
        global SAVE_PATH
        SAVE_PATH = os.path.join(SAVE_PATH, f'alpha_{alpha}')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH, exist_ok=True)

    print("Loading model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    padding = 'max_length'
    glue = ['sst2', 'mnli']

    if cache_dir == 'None':
        cache_dir = None

    if val_file is None:
        if 'glue' in dataset_name:
            dataset_name, dataset_config_name = dataset_name.split('_')
        elif dataset_name in glue:
            dataset_name, dataset_config_name = 'glue', dataset_name

        dataset = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)
        if dataset_config_name is not None:
            task_name = dataset_config_name
        else:
            task_name = dataset_name
        dataloader = process_hf_dataset(dataset, split, task_name, tokenizer, padding, max_length=max_length, batch_size=batch_size, n=n, shuffle=False)
        with_labels = True
    else:
        # Check for file type and process either .tsv or .txt
        if '.tsv' in val_file:
            df = pd.read_table(val_file)
            label_key = 'label'
            if 'mnli' in val_file:
                task_name = 'mnli'
                label_key = 'label'
            elif 'imdb' in val_file:
                task_name = 'counterfactual-imdb'
                label_key = 'Sentiment'
            else: #TODO: Support other tasks
                task_name = 'none'
                return

            if label_key in df:
                with_labels = True
                df = df[df[label_key] != -1]
            else:
                with_labels = False
            # num_labels = len(np.unique(pd.Categorical(df['label'], ordered=True)))
            dataloader = process_custom_dataset(df, task_name, tokenizer, padding, max_length, batch_size, n=n, shuffle=False)
        else:
            dataloader = process_lm_dataset(val_file, tokenizer, padding, max_length, batch_size, n=n, num_label_chars=0, shuffle=False)
            with_labels = False

    print('Evaluating model')
    start_time = time.time()
    probs = eval(model, dataloader, device, with_labels=with_labels)
    end_time = time.time()
    print("MSP runtime:", end_time - start_time)
    np.save(os.path.join(SAVE_PATH, f'{fname}_probs'), probs)
    if save_msp:
        msp = np.max(probs, axis=1)
        np.save(os.path.join(SAVE_PATH, f'{fname}_msp'), msp)

if __name__ == '__main__':
    fire.Fire(main)
    print("\n\n--------DONE--------")
