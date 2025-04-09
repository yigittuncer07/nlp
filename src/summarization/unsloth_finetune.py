#! /usr/bin/env python
# -*- coding: utf-8 -*-

from unsloth import FastLanguageModel
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

MODEL_NAME = "google/gemma-3-1b-it"
DATASET_NAME = "turkishnlp/turkish_summarization"
EVAL_DATASET_NAME = "turkishnlp/turkish_summarization"
TRAIN_ON_COMPLETION_ONLY = False
HF_URL="" # "your_name/lora_model" # Set this to push to HuggingFace Hub

SAVE_PATH = "/home/nlp/projects/nlp/artifacts/models"
SAVE_PATH = f"SAVE_PATH/{MODEL_NAME.split("/")[-1]}"
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def format_and_filter(example):
    # Step 1: Format
    convo = {
        "conversations": [
            {"role": "system", "content": f"{example['instruction']}"},
            {"role": "user", "content": f"{example['input']}"},
            {"role": "assistant", "content": example["output"]}
        ]
    }

    # Step 2: Apply chat template
    text = tokenizer.apply_chat_template(
        convo["conversations"],
        tokenize=False,
        add_generation_prompt=False
    )

    # Step 3: Filter based on tokenized length
    tokenized = tokenizer(text, truncation=False, return_length=True)
    if tokenized["length"][0] < max_seq_length - 3:
        return {"text": text}
    else:
        return {}  # filter

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.shuffle(seed=2523)
dataset = dataset.select(range(0, 100))
dataset = dataset.map(format_and_filter, batched=False)
dataset = dataset.filter(lambda x: "text" in x)  # Remove entries filtered out

print(dataset[5]["text"])

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 16,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 2,
        warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = SAVE_PATH,
        report_to = "none", # Use this for WandB etc
        save_total_limit = 3,
        save_strategy = "steps",
        save_steps = 1000,
    ),
)

if TRAIN_ON_COMPLETION_ONLY:
    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    """We verify masking is actually done:"""

    print(tokenizer.decode(trainer.train_dataset[5]["input_ids"]))

    space = tokenizer(" ", add_special_tokens = False).input_ids[0]
    print(tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]]))

    """We can see the System and Instruction prompts are successfully masked!"""

trainer_stats = trainer.train()

model.save_pretrained(f"{SAVE_PATH}/final")  
tokenizer.save_pretrained(f"{SAVE_PATH}/final")
model.save_pretrained_merged(f"{SAVE_PATH}/final_merged", tokenizer, save_method = "merged_16bit")

if HF_URL:
    model.push_to_hub("your_name/lora_model") # Online saving
    tokenizer.push_to_hub("your_name/lora_model") # Online saving
    model.push_to_hub_merged("hf/model", tokenizer, save_method = "merged_16bit")
