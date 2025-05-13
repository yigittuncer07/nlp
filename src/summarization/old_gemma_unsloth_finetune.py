#! /usr/bin/env python
# -*- coding: utf-8 -*-

# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=yqxqAZ7KJ4oL

from unsloth import FastModel
from datasets import load_dataset, load_from_disk
import torch
import time

MODEL_NAME = "unsloth/gemma-3-4b-it" # CHANGE THIS
DATASET_NAME = "/home/yigit/nlp/artifacts/datasets/summarization-ykd-data" # CHANGE THIS
RUN_NAME = "gemma3-4b-it-ykd-sft-2-epoch-attempt2" # CHANGE THIS
TRAIN_ON_COMPLETIONS = False # CHANGE THIS

SAVE_PATH = "/home/yigit/nlp/artifacts/models"
SAVE_PATH = f"{SAVE_PATH}/{RUN_NAME}"
SYSTEM_PROMPT = "Sen yardımcı bir asistansın, verilen metni özetle."

WANDB_ENABLED = True
if WANDB_ENABLED:
    import wandb
    wandb.init(project="huggingface", name=RUN_NAME)


max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 32,           # Larger = higher accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

def format_example(example):
    formatted = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{example['input']}"},
        {"role": "assistant", "content": example["output"]}
    ]
    
    text = tokenizer.apply_chat_template(
        formatted,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

def is_within_max_length(example):
    tokenized = tokenizer(example["text"], truncation=False, return_length=True)
    return tokenized["length"][0] < max_seq_length - 3

if DATASET_NAME.startswith(".") or DATASET_NAME.startswith("/"):
    dataset = load_from_disk(DATASET_NAME)
else:
    dataset = load_dataset(DATASET_NAME)

train_dataset = dataset["train"]
eval_dataset = dataset["valid"]
train_dataset = train_dataset.shuffle(seed=2523)
# train_dataset = train_dataset.select(range(0, 100))
train_dataset = train_dataset.map(format_example, batched=False)
train_dataset = train_dataset.filter(is_within_max_length)

eval_dataset = eval_dataset.map(format_example, batched=False)
eval_dataset = eval_dataset.filter(is_within_max_length)

print(train_dataset[5]["text"])

def formatting_func(example):
    formatted = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    text = tokenizer.apply_chat_template(
        formatted,
        tokenize=False,
        add_generation_prompt=False
    )
    return text

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = eval_dataset,
    # formatting_func= formatting_func,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 2,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs (2e-4 originally)
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb" if WANDB_ENABLED else "none",
        save_total_limit = 2,
        save_strategy = "epoch",
        # do_eval = True,
        # eval_strategy = "epoch",
        output_dir = SAVE_PATH,
    ),
)

print(f"Starting training for: {MODEL_NAME}")
print(f"Training on {DATASET_NAME}")
print(f"Training on {len(train_dataset)} examples")
print(f"Evaluating on {len(eval_dataset)} examples")
print(f'Will save to {SAVE_PATH}')

time.sleep(3)
trainer_stats = trainer.train()

model.save_pretrained(f"{SAVE_PATH}/final")  
tokenizer.save_pretrained(f"{SAVE_PATH}/final")
model.save_pretrained_merged(f"{SAVE_PATH}/final_merged", tokenizer, save_method = "merged_16bit")