# Import Libraries
import argparse
import fasttext
import evaluate
import gc
import json
import os
import torch
import transformers

import pandas as pd

from datasets import load_dataset, concatenate_datasets
from huggingface_hub import hf_hub_download
from math import ceil
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from prodigyopt import Prodigy
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
)
import wandb

import os
os.environ["WANDB_MODE"]="offline"

# Parse Arguments
print('Parsing arguments...')
parser = argparse.ArgumentParser(
    prog='Training Script',
    description='Handles Training',
)

parser.add_argument('-bm', '--base_model', default='google/mt5-base', type=str, dest='base_model')
parser.add_argument('-tm', '--training_mode', default='fft', type=str, choices=['fft', 'lora', 'qlora'], dest='training_mode')
parser.add_argument('-ln', '--level_names', nargs='+', type=str, choices=['level0', 'level1', 'level2', 'language'], required=True, dest='level_names')
parser.add_argument('-sv', '--skip_values', default=[], nargs='*', type=str, dest='skip_values')
parser.add_argument('-le', '--level_epochs', default=3, type=int, dest='level_epochs')
parser.add_argument('-ls', '--lora_size', default=256, type=int, dest='lora_size')
parser.add_argument('-lf', '--language_families', type=str, required=True, dest='language_families')
parser.add_argument('-bs', '--batch_size', default=8, type=int, choices=[1, 2, 4, 8, 16, 32], dest='batch_size')
parser.add_argument('-ts', '--test_size', default=5000, type=int, dest='test_size')
parser.add_argument('-ge', '--global_evaluations', type=str, dest='global_evaluations')
parser.add_argument('-sn', '--save_name', default='test', type=str, dest='save_name')


args = parser.parse_args()
args.__setattr__('readable_base_model', args.base_model.replace('-','_').replace('/','_'))

# Prepare Custom Callback
class ClearRamCallback(TrainerCallback):
    def __init__(self, reserved_threshold=1_000_000_000, allocated_threshold=10_000_000):
        self.reserved_threshold = reserved_threshold
        self.allocated_threshold = allocated_threshold

    def on_evaluate(self, args, state, control, **kwargs):
        # Clear vram cache only when running out of space
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)
        if t-r < self.reserved_threshold or r-a < self.allocated_threshold:
            print('Emptying cache at global step {state.global_step}')
            print(t,r,a)
            torch.cuda.empty_cache()
            t = torch.cuda.get_device_properties(0).total_memory
            r = torch.cuda.memory_reserved(0)
            a = torch.cuda.memory_allocated(0)
            print(t,r,a)

# Load Tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    args.base_model,
    #cache_dir='.cache'
)
tokenizer.pad_token = tokenizer.eos_token


# Load Language Families
print('Loading language damilies...')
language_families=pd.read_csv(args.language_families)
lang_mapping = {row.code: row.language for index, row in language_families.iterrows()}


# Prepare Evaluations
print('Preparing evaluation...')
sacrebleu_evaluator = evaluate.load('sacrebleu')
chrfpp_evaluator = evaluate.load('chrf')
lid_model_path = hf_hub_download(
    repo_id='cis-lmu/glotlid',
    filename='model.bin',
    #cache_dir='.cache'
)
lid_model = fasttext.load_model(lid_model_path)

def compute_lid(predictions, code):
    predictions_lids = lid_model.predict(predictions, k=3000)
    predictions_lang_indexes = [langs.index(f'__label__{code}') for langs in predictions_lids[0]]
    predictions_scores = [predictions_lids[1][i][index] for i,index in enumerate(predictions_lang_indexes)]
    average_lid = sum(predictions_scores)/len(predictions_scores)
    return average_lid*100

def compute_evaluation(dataset, code):
    if dataset == 'ldc2020t07':
        test_ds = load_dataset(f'data/ldc2020t07', code, split='test', trust_remote_code=True)
        test_ds = test_ds.map(collate_seq2seq_test, remove_columns=test_ds.column_names)
    if dataset=='flores':
        test_ds = load_dataset(f'data/SilverAMR_GoldText', f'flores.{code}', split='test', trust_remote_code=True)
        test_ds = test_ds.map(collate_seq2seq, remove_columns=test_ds.column_names)

    test_dl = DataLoader(test_ds, collate_fn=DataCollatorForSeq2Seq(tokenizer), batch_size=args.batch_size, shuffle=False)

    references = []
    predictions = []

    with torch.inference_mode():
        for batch in tqdm(test_dl):
            outputs = model.generate(
                input_ids=batch['input_ids'].to('cuda'),
                attention_mask=batch['attention_mask'].to('cuda'),
                max_new_tokens=128
            )
            labels = batch['labels']
            labels[labels == -100] = tokenizer.pad_token_id
            references += tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions += tokenizer.batch_decode(outputs, max_new_tokens=128, skip_special_tokens=True)

    old_bleu = sacrebleu_evaluator.compute(predictions=predictions, references=references)['score']
    bleu = sacrebleu_evaluator.compute(tokenize='flores200', predictions=predictions, references=references)['score']
    chrfpp = chrfpp_evaluator.compute(word_order=2, predictions=predictions, references=references)['score']
    lid = compute_lid(predictions=predictions, code=row.code)
    lid_diff = lid - global_evaluations[dataset]['lid'][row.code]

    return old_bleu, bleu, chrfpp, lid, lid_diff, predictions, references


# Computer Global Evaluations
print('Computing global evaluations')
if args.global_evaluations is not None:
    with open(args.global_evaluations, 'r') as file:
        global_evaluations = json.load(file)

if not global_evaluations:
    global_evaluations = {'flores':{'lid':{}}, 'ldc2020t07':{'lid':{},}}

for code in tqdm(language_families.code):
    print(f' * {code}')
    data_source = 'GoldAMR_SilverText'
    if code == 'eng_Latn':
        data_source = 'GoldAMR_GoldText'
    temp_ds = load_dataset(f'data/{data_source}', f'amr3.{code}', trust_remote_code=True)
    temp_ds = load_dataset(f'data/SilverAMR_GoldText', f'flores.{code}', trust_remote_code=True)
    if code not in global_evaluations['flores']['lid']:
        global_evaluations['flores']['lid'][code] = compute_lid(predictions=temp_ds['test']['target'], code=code)
    print('    - Flores LID: '+str(global_evaluations['flores']['lid'][code]))
    if code in ['deu_Latn', 'eng_Latn', 'ita_Latn', 'spa_Latn']:
        temp_ds = load_dataset(f'data/ldc2020t07', code, trust_remote_code=True)
        if code not in global_evaluations['ldc2020t07']['lid']:
            global_evaluations['ldc2020t07']['lid'][code] = compute_lid(predictions=temp_ds['test']['text'], code=code)
        print('    - LDCT2020T07 LID: '+str(global_evaluations['ldc2020t07']['lid'][code]))
    with open(args.global_evaluations, 'w') as file:
        json.dump(global_evaluations, file, indent=4)
print(global_evaluations)


# Prepare Data Collators
print('Preparing data collators...')
def collate_seq2seq(sample):

    source_text = sample['source']
    target_lang_code = sample['target_lang']
    target_lang_name = lang_mapping[target_lang_code]
    target_text = sample['target']

    input_text = f'Translate to {target_lang_name}: {source_text}'

    tokenized_input = tokenizer(input_text)
    tokenized_output = tokenizer(target_text)

    tokenized_input['labels'] = tokenized_output['input_ids']

    return tokenized_input

def collate_seq2seq_test(sample):

    source_text = sample['graph']
    target_lang_code = sample['lang']
    target_lang_name = lang_mapping[target_lang_code]
    target_text = sample['text']

    input_text = f'Translate to {target_lang_name}: {source_text}'

    tokenized_input = tokenizer(input_text)
    tokenized_output = tokenizer(target_text)

    tokenized_input['labels'] = tokenized_output['input_ids']

    return tokenized_input

# Prepare Q-LoRA Configuration
print('Preparing Q-LoRA configuration...')
peft_config = LoraConfig(
    r=args.lora_size,
    lora_alpha=args.lora_size,
    lora_dropout=0.05,
    target_modules='all-linear',
    use_rslora=True,
    bias="none",
    task_type="SEQ_2_SEQ_LM",
)

# Training Loop
print('Training Loop...')
for level, level_name in enumerate(tqdm(args.level_names)):
    for level_value in tqdm(language_families[level_name].dropna().unique(), position=1):
        if level_value not in args.skip_values:
            wandb.init(project=f'{args.save_name}_{args.readable_base_model}', group=f'{args.training_mode}_{level_name}_{level_value}', entity='williamsm')
            # Prepare Data
            print('Preparing data...')
            temp_language_families = language_families[language_families[level_name]==level_value].reset_index(drop=True)
            all_train_ds = []
            all_eval_ds = []
            eval_per_lang = args.test_size // len(temp_language_families) // 2
            for index, row in tqdm(temp_language_families.iterrows(), total=len(temp_language_families), position=2):
                data_source = 'GoldAMR_SilverText'
                if row.code == 'eng_Latn':
                    data_source = 'GoldAMR_GoldText'
                temp_ds = load_dataset(
                    f'data/{data_source}',
                    f'amr3.{row.code}',
                    trust_remote_code=True,
                    #cache_dir='.cache'
                )
                all_train_ds.append(temp_ds['train'])
                all_eval_ds.append(temp_ds['validation'].select(range(min(len(temp_ds['validation']), eval_per_lang))))
                temp_ds2 = load_dataset(
                    f'data/SilverAMR_GoldText',
                    f'flores.{row.code}',
                    trust_remote_code=True,
                    #cache_dir='.cache'
                )
                all_train_ds.append(temp_ds2['train'])
                all_eval_ds.append(temp_ds2['validation'].select(range(min(len(temp_ds2['validation']), eval_per_lang))))
            train_ds = concatenate_datasets(all_train_ds).map(collate_seq2seq)
            eval_ds = concatenate_datasets(all_eval_ds).map(collate_seq2seq)


            # Prepare Base Model
            print('Preparing model...')
            if args.training_mode == 'fft':
                if level <= 0:
                    model = MT5ForConditionalGeneration.from_pretrained(
                        args.base_model,
                        low_cpu_mem_usage=True,
                        #cache_dir='.cache',
                    )
                else:
                    previous_level = level-1
                    previous_level_name = f'level{previous_level}'
                    previous_level_value = temp_language_families[previous_level_name].dropna().unique()[0]
                    load_path = f'{args.save_name}_{args.readable_base_model}_{args.training_mode}_{previous_level_name}_{previous_level_value}'
                    model = MT5ForConditionalGeneration.from_pretrained(
                        load_path,
                        low_cpu_mem_usage=True,
                        #cache_dir='.cache',
                    )
            else:
                if args.training_mode == 'qlora':
                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type='nf4',
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    model = MT5ForConditionalGeneration.from_pretrained(
                        args.base_model,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        #quantization_config=nf4_config,
                        load_in_4bit=True,
                        #cache_dir='.cache',
                    )
                else:
                    model = MT5ForConditionalGeneration.from_pretrained(
                        args.base_model,
                        low_cpu_mem_usage=True,
                        #cache_dir='.cache',
                    )

                if level > 0:
                    for previous_level in range(0, level):
                        previous_level_name = f'level{previous_level}'
                        previous_level_value = temp_language_families[previous_level_name].dropna().unique()[0]
                        model = PeftModel.from_pretrained(
                            model,
                            f'{args.save_name}_{args.readable_base_model}_{args.training_mode}/{previous_level_name}_{previous_level_value}',
                            #cache_dir='.cache',
                        )
                        model = model.merge_and_unload()

                model.config.use_cache = False
                model.gradient_checkpointing_enable()
                model = prepare_model_for_kbit_training(model)
                adapter_name = f'{level_name}_{level_value}'
                model = get_peft_model(model, peft_config, adapter_name=adapter_name)
                model.set_adapter(adapter_name)
                # model = model.to_bettertransformer() # Not supported yet
                model.print_trainable_parameters()

            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()

            eval_logging_save_steps = 500

            # Seq2Seq Training Arguments
            print('Setting arguments...')
            seq2seq_training_args = Seq2SeqTrainingArguments(
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                num_train_epochs=args.level_epochs,
                save_strategy='steps',
                evaluation_strategy='steps',
                logging_strategy='steps',
                save_steps=eval_logging_save_steps,
                eval_steps=eval_logging_save_steps,
                logging_steps=eval_logging_save_steps,
                save_total_limit=3,
                load_best_model_at_end=True,
                output_dir=f'temp',
                report_to="wandb",
                optim='adafactor',
                prediction_loss_only=True,
                group_by_length=True,
                save_safetensors=False,
                hub_strategy='end',
            )

            #optimizer = Prodigy(model.parameters(), weight_decay=0.1, decouple=True)
            #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
            #scheduler=None

            # Seq2Seq Trainer
            print('Setting trainer...')
            seq2seq_trainer = Seq2SeqTrainer(
                model,
                args=seq2seq_training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=DataCollatorForSeq2Seq(tokenizer),
                tokenizer=tokenizer,
                #optimizers=(optimizer, scheduler),
                callbacks = [ClearRamCallback()],
                #callbacks = [EarlyStoppingCallback(early_stopping_patience = 5), ClearRamCallback()],
                #callbacks = [EarlyStoppingCallback(early_stopping_patience = 5)],
            )

            # Train
            print('Training...')
            os.environ['HF_HUB_OFFLINE']='1'
            os.environ['TRANSFORMER_OFFLINE']='1'
            seq2seq_trainer.train()

            # Save
            print('Saving...')
            if args.training_mode == 'fft':
                save_path = f'{args.save_name}_{args.readable_base_model}_{args.training_mode}_{level_name}_{level_value}'
            else:
                save_path = f'{args.save_name}_{args.readable_base_model}_{args.training_mode}'

            seq2seq_trainer.model.save_pretrained(save_path, safe_serialization=False)
            tokenizer.save_pretrained(save_path)
            os.environ['HF_HUB_OFFLINE']='0'
            os.environ['TRANSFORMER_OFFLINE']='0'

            # Evaluation
            print('Evaluating...')
            if 'lora' in args.training_mode:
                model = seq2seq_trainer.model.merge_and_unload()
            for index, row in tqdm(temp_language_families.iterrows(), total=len(temp_language_families), position=2):
                print(row.code)
                evaluation = {}
                if row.code in ['deu_Latn', 'eng_Latn', 'ita_Latn', 'spa_Latn']:
                    print(' * ldc2020t07')
                    old_bleu, bleu, chrfpp, lid, lid_diff, predictions, references = compute_evaluation('ldc2020t07', row.code)
                    evaluation['ldc2020t07'] = {'old_bleu':old_bleu, 'bleu':bleu, 'chrf++':chrfpp, 'lid':lid, 'lid_diff':lid_diff, 'predictions':predictions, 'references':references}
                print(' * flores')
                old_bleu, bleu, chrfpp, lid, lid_diff, predictions, references = compute_evaluation('flores', row.code)
                evaluation['flores'] = {'old_bleu':old_bleu, 'bleu':bleu, 'chrf++':chrfpp, 'lid':lid, 'lid_diff':lid_diff, 'predictions':predictions, 'references':references}

                print(' * saving evaluation...')
                os.makedirs(f'{save_path}/evaluation', exist_ok=True)
                with open(f'{save_path}/evaluation/{level_name}_{level_value}_{row.code}.json', 'w', encoding='utf-8') as file:
                    json.dump(evaluation, file, indent=4)
                wandb.finish()
