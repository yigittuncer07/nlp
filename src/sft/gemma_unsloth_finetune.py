#! /usr/bin/env python
# -*- coding: utf-8 -*-

# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb#scrollTo=yqxqAZ7KJ4oL

from unsloth import FastModel
from datasets import load_dataset, load_from_disk
import time
import json

MODEL_NAME = "/media/drive1/gemma-3-4b-it-mevzuat-ders_kitaplari-cpt/final_merged" # CHANGE THIS
# MODEL_NAME = "google/gemma-3-4b-it"
DATASET_NAME = "/mnt/2tbdrive/shared_folder/datasets/question-answer-cr-4o-clean-sft/" # CHANGE THIS
# RUN_NAME = "gemma-3-4b-it-question-answer-cr-4o-clean-sft" # CHANGE THIS
RUN_NAME = "gemma-3-4b-it-mevzuat-ders_kitaplari-cpt-question-answer-cr-4o-clean-sft" # CHANGE THIS
TRAIN_ON_COMPLETIONS = False # CHANGE THIS
r = 128 * 4

SAVE_PATH = "/media/drive1"
SAVE_PATH = f"{SAVE_PATH}/{RUN_NAME}"

WANDB_ENABLED = False
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

    r = r,           # Larger = higher accuracy, but might overfit
    lora_alpha = r * 2,  # Recommended alpha == r at least
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
    if isinstance(example["messages"], str):
        example["messages"] = json.loads(example["messages"])
    
    text = tokenizer.apply_chat_template(
        example["messages"],
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
# eval_dataset = dataset["valid"]
train_dataset = train_dataset.shuffle(seed=2523)
# train_dataset = train_dataset.select(range(0, 100))
train_dataset = train_dataset.map(format_example, batched=False)
train_dataset = train_dataset.filter(is_within_max_length)

# eval_dataset = eval_dataset.map(format_example, batched=False)
# eval_dataset = eval_dataset.filter(is_within_max_length)

print(train_dataset[5]["text"])

per_device_train_batch_size = 1
gradient_accumulation_steps = 1 
epochs = 3

steps_per_epoch = len(train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    # eval_dataset = eval_dataset,
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps, # Use GA to mimic batch size!
        per_device_eval_batch_size = per_device_train_batch_size,
        eval_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = epochs,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs (2e-4 originally)
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "wandb" if WANDB_ENABLED else "none",
        save_total_limit = epochs,
        save_strategy = "epoch",
        # do_eval = False,
        # eval_strategy = "steps",
        # eval_steps = steps_per_epoch // 5,
        # eval_on_start = True,
        output_dir = SAVE_PATH,
    ),
)

print(f"Starting training for: {MODEL_NAME}")
print(f"Training on {DATASET_NAME}")
print(f"Training on {len(train_dataset)} examples")
# print(f"Evaluating on {len(eval_dataset)} examples")
print(f'Will save to {SAVE_PATH}')

time.sleep(7)

model.config.text_config.use_cache = False
trainer.model.config.use_cache = False

if TRAIN_ON_COMPLETIONS:
    print("TRAINING ON COMPLETIONS ONLY!")
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )
    print('training on completions, train example:')
    tokenizer.decode(trainer.train_dataset[100]["input_ids"])
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))
    
    
    print("training on completions, eval example:")
    tokenizer.decode(trainer.eval_dataset[100]["input_ids"])
    print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.eval_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))

# with torch.autograd.detect_anomaly(True):
trainer_stats = trainer.train(resume_from_checkpoint = True)
    
model.save_pretrained(f"{SAVE_PATH}/final")  
tokenizer.save_pretrained(f"{SAVE_PATH}/final")
model.save_pretrained_merged(f"{SAVE_PATH}/final_merged", tokenizer, save_method = "merged_16bit")