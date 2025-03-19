from unsloth import FastModel
import torch
from datasets import load_dataset, load_from_disk
import wandb

MODEL_NAME="/home/yigittuncer/nlp/playground/gemma-4-cpt"
OUTPUT_DIR=f"/media/drive1/{MODEL_NAME.split('/')[-1]}-sft"
DATASET_NAME="/media/drive1/question-answer-cr-4o-clean-sft" 
WANDB=True
if WANDB:
    wandb.login()
    config = {
        "model_name": MODEL_NAME,
    }
    wandb.init(project="SFT", name=MODEL_NAME, config=config)

model, tokenizer = FastModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

"""We now add LoRA adapters so we only need to update a small amount of parameters!"""

# model = FastModel.get_peft_model(
#     model,
#     finetune_vision_layers     = False, # Turn off for just text!
#     finetune_language_layers   = True,  # Should leave on!
#     finetune_attention_modules = True,  # Attention good for GRPO
#     finetune_mlp_modules       = True,  # SHould leave on always!

#     r = 8,           # Larger = higher accuracy, but might overfit
#     lora_alpha = 8,  # Recommended alpha == r at least
#     lora_dropout = 0,
#     bias = "none",
#     random_state = 3407,
# )

"""
    model = FastModel.get_peft_model(
            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/yigittuncer/nlp/.unslothvenv/lib/python3.12/site-packages/unsloth/models/vision.py", line 422, in get_peft_model
    raise RuntimeError("Unsloth: You already added LoRA adapters to your model!")
RuntimeError: Unsloth: You already added LoRA adapters to your model!
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

if DATASET_NAME.startswith("./") or DATASET_NAME.startswith("/"):
    dataset=load_from_disk(DATASET_NAME)
    dataset=dataset["train"]
else:
    dataset=load_dataset(DATASET_NAME, split = "train")

"""We now use `standardize_data_formats` to try converting datasets to the correct format for finetuning purposes!"""

from unsloth.chat_templates import standardize_data_formats
dataset = standardize_data_formats(dataset)

import json
"""We now have to apply the chat template for `Gemma-3` onto the conversations, and save it to `text`"""

def apply_chat_template(examples):
    formatted_texts = []
    
    for message_json in examples["messages"]:
        try:
            # Parse the message string into a Python object
            messages = json.loads(message_json)

            # Ensure messages alternate correctly
            for i in range(len(messages) - 1):
                if messages[i]["role"] == messages[i + 1]["role"]:
                    raise ValueError("Conversation roles must alternate.")

            # Apply chat template
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_texts.append(text)
        except Exception as e:
            print(f"Skipping invalid conversation due to error: {e}")
            formatted_texts.append("")  # Handle errors gracefully

    return {"text": formatted_texts}
pass
dataset = dataset.map(apply_chat_template, batched = True)

"""Let's see how the chat template did! Notice `Gemma-3` default adds a `<bos>`!"""

print(dataset[100]["text"])

"""<a name="Train"></a>
### Train the model
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer). We do 60 steps to speed things up, but you can set `num_train_epochs=1` for a full run, and turn off `max_steps=None`.
"""

from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = None,
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        report_to = "wandb" if WANDB else None,
        save_total_limit = 3,
        save_strategy = "steps",
        save_steps = 1000,
    ),
)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

"""Let's verify masking the instruction part is done! Let's print the 100th row again:"""

print(tokenizer.decode(trainer.train_dataset[100]["input_ids"]))

"""Now let's print the masked out example - you should see only the answer is present:"""

print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[100]["labels"]]).replace(tokenizer.pad_token, " "))

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

trainer_stats = trainer.train()

"""<a name="Inference"></a>
### Inference
Let's run the model via Unsloth native inference! According to the `Gemma-3` team, the recommended settings for inference are `temperature = 1.0, top_p = 0.95, top_k = 64`
"""

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)
messages = [{
    "role": "user",
    "content": [{
        "type" : "text",
        "text" : "Continue the sequence: 1, 1, 2, 3, 5, 8,",
    }]
}]
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)
outputs = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 64, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 1.0, top_p = 0.95, top_k = 64,
)
print(tokenizer.batch_decode(outputs))

"""<a name="Save"></a>
### Saving, loading finetuned models
To save the final model as LoRA adapters, either use Huggingface's `push_to_hub` for an online save or `save_pretrained` for a local save.

**[NOTE]** This ONLY saves the LoRA adapters, and not the full model. To save to 16bit or GGUF, scroll down!
"""

model.save_pretrained(f"{OUTPUT_DIR}/final")  # Local saving
tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
# model.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving
# tokenizer.push_to_hub("HF_ACCOUNT/gemma-3", token = "...") # Online saving

if False: # Change to True to save finetune!
    model.save_pretrained_merged("gemma-3-finetune", tokenizer)