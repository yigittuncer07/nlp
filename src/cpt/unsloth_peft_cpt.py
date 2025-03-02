from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
import warnings
warnings.filterwarnings("ignore", message=".*average_tokens_across_devices.*")

#!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth.git
#export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1


max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

OUTPUT_DIR = "../../artifacts/models/llama3.2-1b-instruct_cpt-FULL"
REPO_NAME = "turkishnlp/Llama-3.2-1B-Instruct-CPT-oscar-Unsloth-FULL"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",

                      "embed_tokens", "lm_head",], # Add for continual pretraining
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = True,   # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    texts  = examples["text"]
    outputs = []
    
    for text in texts:
        tokenized = tokenizer(text, truncation=False, return_length=True)
        if tokenized["length"][0] < max_seq_length - 3:
            outputs.append(text + EOS_TOKEN)  # Ensure EOS token is added only for valid sequences
    
    return { "text": outputs }


dataset = load_dataset("musabg/wikipedia-oscar-tr", split = "train",)

dataset = dataset.train_test_split(train_size = 0.1)["train"]

dataset = dataset.map(formatting_prompts_func, batched = True,)

trainer = UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 8,

    args = UnslothTrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 8,

        # Use warmup_ratio and num_train_epochs for longer runs!
        warmup_ratio = 0.1,
        num_train_epochs = 1,

        # Select a 2 to 10x smaller learning rate for the embedding matrices!
        learning_rate = 5e-5,
        embedding_learning_rate = 1e-5,

        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 42,
        output_dir = OUTPUT_DIR,
        report_to = "none", # Use this for WandB etc
    ),
)

try:
    trainer_stats = trainer.train()
except:
    model.save_pretrained(f"{OUTPUT_DIR}/final")
    model.push_to_hub(REPO_NAME)
    breakpoint()

model.save_pretrained(f"{OUTPUT_DIR}/final") 
try:
    model.push_to_hub(REPO_NAME)
except Exception as e:
    breakpoint()
