from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import fire
import time
import pandas as pd

SAVE_PATH = 'output/'
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

def compute_all(model, all_encodings, fname, save_path=SAVE_PATH, device='cpu'):
    start_time = time.time()
    print(f"Finding perplexities for {fname}")
    perplexities, lls = [], []
    # all_encodings = all_encodings[:n]
    pbar = tqdm(total=len(all_encodings))
    for idx, encodings in enumerate(all_encodings):
        try:
            pp, ll = compute_perplexity(model, encodings, device=device)
            perplexities.append(pp)
            lls.append(ll)
        except Exception as e:
            print("Exception at idx", idx)
            print(e)
            continue
        finally:
            pbar.update(1)
    
    pbar.close()
    end_time = time.time()
    print("PPL runtime:", end_time-start_time)

    perplexities = np.array(perplexities)
    np.save(f'{save_path}/{fname}_pps.npy', perplexities)
    print(f"\nMean: {perplexities.mean()}, Std: {perplexities.std()}")

    with open(f'{save_path}/{fname}_lls.pkl', 'wb') as fw:
        pickle.dump(lls, fw)

    return perplexities

def compute_perplexity(model, encodings, stride=None, device='cuda'):
    max_length = model.config.n_positions
    lls = []
    if stride is None:
        # stride = max(1, encodings.input_ids.size(1) // 100)
        # stride = max_length
        # print('Stride:', stride)
        stride = 1

    for i in range(1, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i)
    return ppl.item(), torch.stack(lls).detach().cpu().numpy()

def setup(path='lvwerra/gpt2-imdb'):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    n = None
    model = AutoModelWithLMHead.from_pretrained(path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer, device, n

def get_encoding(tokenizer, text, label=None, add_eot=True):
    if label is not None:
        prefix = f'{label} '
    else:
        prefix = ''
    if add_eot:
        return tokenizer(prefix + text + ' <|endoftext|>', return_tensors='pt')
    else:
        return tokenizer(prefix + text, return_tensors='pt')

def process_hf_dataset(dataset_name, model, tokenizer, device, n=None, key='text', configs=None, fname=None, conditional=False):
    print(f"\n------Processing perplexity for dataset: {dataset_name}-------")

    if configs is None:
        dataset = [load_dataset(dataset_name, split='test')]
    else:
        dataset = [load_dataset(dataset_name, config, split='test') for config in configs]
    
    if fname is None:
        fname = dataset_name

    print("Tokenizing data")
    if not conditional:
        all_encodings = [get_encoding(tokenizer, text) for _dataset in dataset for text in _dataset[key][:n]]
        return compute_all(model, all_encodings, fname)
    else:
        all_encodings_0 = [get_encoding(tokenizer, text, 0) for _dataset in dataset for text in _dataset[key][:n]]
        all_encodings_1 = [get_encoding(tokenizer, text, 1) for _dataset in dataset for text in _dataset[key][:n]]
        fname_0 = fname + '_conditional_0'
        fname_1 = fname + '_conditional_1'

        compute_all(model, all_encodings_0, fname_0)
        compute_all(model, all_encodings_1, fname_1)

def process_entailment(dataset_name, model, tokenizer, device, n=None, dataset_subname=None, fname=None, key1='premise', key2='hypothesis', conditional=False):
    print(f"\n------Processing perplexity for dataset: {dataset_name}_{dataset_subname}-------")

    if dataset_subname is None:
        dataset = load_dataset(dataset_name, split='validation')
    else:
        dataset = load_dataset(dataset_name, dataset_subname, split='validation')
    dataset_texts = []
    for ex in dataset:
        dataset_texts.append(ex[key1] + ' ' + ex[key2])

    if fname is None:
        fname = dataset_name

    print("Tokenizing data")
    if not conditional:
        all_encodings = [get_encoding(tokenizer, text) for text in dataset_texts[:n]]
        return compute_all(model, all_encodings, fname)
    else:
        all_encodings_0 = all_encodings = [get_encoding(tokenizer, text, 0) for text in dataset_texts[:n]]
        all_encodings_1 = all_encodings = [get_encoding(tokenizer, text, 1) for text in dataset_texts[:n]]
        fname_0 = fname + '_conditional_0'
        fname_1 = fname + '_conditional_1'

        compute_all(model, all_encodings_0, fname_0)
        compute_all(model, all_encodings_1, fname_1)

def process_tsv(dataset_path, n=None, key='Text', key2=None):
    print('Loading data...')
    dataset = pd.read_table(dataset_path)
    if key2 is None:
        return dataset[key][:n]
    else:
        series = dataset[key][:n] + ' ' + dataset[key2][:n]
        return series

def process_txt(dataset_path, n=None):
    print('Loading data...')
    with open(dataset_path) as f:
        dataset = f.readlines()
    return dataset[:n]

def process_label(model, tokenizer, device, dataset, label, fname, save_path, n=None):
    print(f"Evaluating conditional for label {label}")
    all_encodings_curr = [get_encoding(tokenizer, text, label, add_eot=False) for text in dataset[:n]]
    fname_curr = fname + f'_conditional_{label}'
    compute_all(model, all_encodings_curr, fname_curr, save_path=save_path, device=device)

def process_dataset(dataset, model, tokenizer, device, fname, save_path, n=None, add_eot=False, conditional=False, num_classes=2, class_num=None):
    if not conditional:
        all_encodings = [get_encoding(tokenizer, text, add_eot=add_eot) for text in dataset[:n]]
        compute_all(model, all_encodings, fname, save_path=save_path, device=device)
    else:
        if class_num is None:
            for label in range(num_classes):
                process_label(model, tokenizer, device, dataset, label, fname, save_path, n=n)
        else:
            process_label(model, tokenizer, device, dataset, class_num, fname, save_path, n=n)

def main(dataset_path, model_path='/scratch/ua388/ckpts/gpt2-glue_sst2', fname=None, n=None, conditional=False, add_eot=False, num_classes=2, class_num=None, key='sentence1', key2='sentence2', num_splits=1, split_idx=None):
    model, tokenizer, device, _ = setup(model_path)
    if conditional:
        save_path = os.path.join(SAVE_PATH, 'gpt2_conditional')
    else:
        save_path = os.path.join(SAVE_PATH, 'gpt2')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if '.txt' in dataset_path:
        dataset = process_txt(dataset_path, n)
        # Split dataset examples
        if split_idx is not None:
            fname = f'{fname}_{split_idx}'
            per_split = len(dataset) // num_splits
            start_idx = split_idx * per_split
            if split_idx + 1 < num_splits:
                end_idx = (split_idx + 1) * per_split
            else:
                end_idx = len(dataset)
            dataset = dataset[start_idx:end_idx]
            print(f"Taking split {split_idx+1}/{num_splits} with start_idx: {start_idx} and end_idx: {end_idx} of size {len(dataset)}")
    elif '.tsv' in dataset_path:
        dataset = process_tsv(dataset_path, n=n, key=key, key2=key2)
    else:
        dataset = None
        print("Invalid dataset path:", dataset_path)
        return

    if '<|endoftext|>' in dataset[0]:
        add_eot = False
    else:
        add_eot = True

    process_dataset(dataset, model, tokenizer, device, fname, save_path, n=n, add_eot=add_eot, conditional=conditional, num_classes=num_classes, class_num=class_num)

if __name__ == '__main__':
    print("Loading model...")

    # To evaluate HF datasets, use these
    # model, tokenizer, device, n = setup('ckpts/gpt2-glue_sst2')
    # process_hf_dataset('imdb', model, tokenizer, device, n=2, key='text', conditional=True)
    # process_hf_dataset('yelp_polarity', model, tokenizer, device, n=3000, key='text')
    # process_hf_dataset('glue', model, tokenizer, device, configs=['sst2'], key='sentence', fname='sst2')
    # process_entailment('glue', model, tokenizer, device, dataset_subname='rte', fname='rte', key1='sentence1', key2='sentence2', conditional=True)
    # process_entailment('snli', model, tokenizer, device, key1='premise', key2='hypothesis', conditional=True)

    fire.Fire(main)

    print("\n\n--------DONE--------")
