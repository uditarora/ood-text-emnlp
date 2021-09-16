# ood-text-emnlp
Code for EMNLP'21 paper "Types of Out-of-Distribution Texts and How to Detect Them"

## Files
- `fine_tune.py` is used to finetune the GPT-2 models, and `roberta_fine_tune.py` is used to finetune the Roberta models.
- `perplexity.py` and `msp_eval.py` is used to find the PPLs and MSPs of a dataset pair's exxamples using the finetuned model.

## How to run
These steps show how to train both density estimation and calibration models on the MNLI dataset, and evaluated against SNLI.

A differet dataset pair can be used by updating the approriate `dataset_name` or `id_data`/`ood_data` values as shown below:

### Training the Density Estimation Model (GPT-2)
Two options:
1. Using HF Datasets -
   ```
   python fine_tune.py --dataset_name glue --dataset_config_name mnli --key premise --key2 hypothesis
   ```
   This also generates a txt train file corresponding to the dataset's text.
2. Using previously generated txt file -
   ```
   python fine_tune.py --train_file data/glue_mnli_train.txt --fname glue_mnli"
   ```

### Finding Perplexity (PPL)
This uses the txt files generated after running `fine_tune.py` to find the perplexity of the ID model on both ID and OOD validation sets -
```
id_data="glue_mnli"
ood_data="snli"
python perplexity.py --model_path ckpts/gpt2-$id_data/ --dataset_path data/${ood_data}_val.txt --fname ${id_data}_$ood_data

python perplexity.py --model_path ckpts/gpt2-$id_data/ --dataset_path data/${id_data}_val.txt --fname ${id_data}_$id_data
```

### Training the Calibration Model (RoBERTa)
Two options:
1. Using HF Datasets -
   ```
   id_data="mnli"
   python roberta_fine_tune.py --task_name $id_data --output_dir /scratch/ua388/roberta_ckpts/roberta-$id_data --fname ${id_data}_$id_data
   ```

2. Using txt file generated earlier -
   ```
   id_data="mnli"
   python roberta_fine_tune.py --train_file data/mnli/${id_data}_conditional_train.txt --val_file data/mnli/${id_data}_val.txt --output_dir roberta_ckpts/roberta-$id_data --fname ${id_data}_$id_data"
   ```
   The `*_conditional_train.txt` file contains both the labels as well as the text.

### Finding Maximum Softmax Probability (MSP)
Two options:
1. Using HF Datasets -
   ```
   id_data="mnli"
   ood_data="snli"
   python msp_eval.py --model_path roberta_ckpts/roberta-$id_data --dataset_name $ood_data --fname ${id_data}_$ood_data
   ```
2. Using txt file generated earlier -
   ```
   id_data="mnli"
   ood_data="snli"
   python msp_eval.py --model_path roberta_ckpts/roberta-$id_data --val_file data/${ood_data}_val.txt --fname ${id_data}_$ood_data --save_msp True
   ```

### Evaluating AUROC
1. Compute AUROC of PPL using `compute_auroc` in `utils.py` -
    ```
    id_data = 'glue_mnli'
    ood_data = 'snli'
    id_pps = utils.read_model_out(f'output/gpt2/{id_data}_{id_data}_pps.npy')
    ood_pps = utils.read_model_out(f'output/gpt2/{id_data}_{ood_data}_pps.npy')
    score = compute_auroc(id_pps, ood_pps)
    print(score)
    ```

2. Compute AUROC of MSP -
   ```
    id_data = 'mnli'
    ood_data = 'snli'
    id_msp = utils.read_model_out(f'output/roberta/{id_data}_{id_data}_msp.npy')
    ood_msp = utils.read_model_out(f'output/roberta/{id_data}_{ood_data}_msp.npy')
    score = compute_auroc(-id_msp, -ood_msp)
    print(score)
   ```