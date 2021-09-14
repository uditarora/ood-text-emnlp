from datasets import load_dataset
import os
import fire

if not os.path.exists('data'):
    os.makedirs('data')

NUM_EPOCHS = 1

CKPT_DIR = 'ckpts'
if not os.path.exists(CKPT_DIR):
    os.makedirs(CKPT_DIR)

def main(train_file=None, val_file=None, dataset_name=None, dataset_config_name=None, key='text', val_name='validation', key2=None, conditional=False, do_val=False, cache_dir='cache/huggingface/transformers', fname=None, version='gpt2'):
    if train_file is None:
        if dataset_config_name is not None:
            fname = dataset_name + '_' + dataset_config_name
        else:
            fname = dataset_name

        if do_val:
            train, val = load_dataset(dataset_name, dataset_config_name, split=['train', val_name], cache_dir=cache_dir)
        else:
            train = load_dataset(dataset_name, dataset_config_name, split='train', cache_dir=cache_dir)

        print(f"Processing dataset {fname}...")
        
        train_str = ""
        for ex in train:
            if conditional:
                line = f"{ex['label']} "
            else:
                line = ""
            line += f"{ex[key]}"
            if key2 is not None:
                line += f" {ex[key2]}"
            train_str += f"{line} <|endoftext|>\n"

        if do_val:
            val_str = ""
            for ex in val:
                if conditional:
                    line = f"{ex['label']} "
                else:
                    line = ""
                line += f"{ex[key]}"
                if key2 is not None:
                    line += f" {ex[key2]}"
                val_str += f"{line} <|endoftext|>\n"

        if conditional:
            fname_train = f'data/{fname}_conditional_train.txt'
            fname_val = f'data/{fname}_conditional_val.txt'
        else:
            fname_train = f'data/{fname}_train.txt'
            fname_val = f'data/{fname}_val.txt'

        with open (fname_train, 'w') as f:
            f.write(train_str)
        
        if do_val:
            with open (fname_val, 'w') as f:
                f.write(val_str)
    else:
        fname_train = train_file
        fname_val = val_file

    print(f"Running fine-tuning from {fname_train}...")

    if conditional == False:
        output_dir = f'--output_dir {CKPT_DIR}/{version}-{fname} '
    else:
        output_dir = f'--output_dir {CKPT_DIR}/{version}-{fname}-conditional '

    if do_val:
        cmd = 'python run_language_modeling.py ' + \
        f'--train_data_file {fname_train} ' + \
        f'--eval_data_file {fname_val} ' + \
        output_dir + \
        f'--model_type {version} ' + \
        f'--model_name_or_path {version} ' + \
        '--save_total_limit 1 ' + \
        f'--num_train_epochs {NUM_EPOCHS} ' + \
        '--do_train \
        --evaluate_during_training \
        --logging_steps 500 \
        --save_steps 500 \
        --do_eval \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 8 \
        --line_by_line \
        --gradient_accumulation_steps 1'
    else:
        cmd = 'python run_language_modeling.py ' + \
        f'--train_data_file {fname_train} ' + \
        output_dir + \
        f'--model_type {version} ' + \
        f'--model_name_or_path {version} ' + \
        '--save_total_limit 1 ' + \
        f'--num_train_epochs {NUM_EPOCHS} ' + \
        '--do_train \
        --per_gpu_train_batch_size 8 \
        --per_gpu_eval_batch_size 8 \
        --line_by_line \
        --gradient_accumulation_steps 1'

    if cache_dir is not None:
        cmd += f' --cache_dir {cache_dir}'

    cmd += ' --overwrite_output_dir'

    os.system(cmd)

if __name__ == '__main__':
    fire.Fire(main)
    print("\n\n--------DONE--------")
