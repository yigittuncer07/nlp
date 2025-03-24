from unsloth import FastLanguageModel
from datasets import load_from_disk
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

max_seq_length=1024*4 # Choose any! We auto support RoPE Scaling internally!
dtype=None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+

MODEL_NAME='turkishnlp/Llama-3.2-1B-Instruct-CPT-oscar-Unsloth-FULL'
OUTPUT_DIR = f"./{MODEL_NAME.split('/')[-1]}-merged"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    dtype = dtype,
)

model.save_pretrained_merged(OUTPUT_DIR, tokenizer)