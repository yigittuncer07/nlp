### HANDLE IMPORTS ###
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig

### SET VARIABLES ###
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATASET = "turkishnlp/conversation_summarization"
MAX_SEQ_LENGTH= 1024
OUTPUT_DIR="../../artifacts/models/llama3.2-1b_summarization"

### PULL MODEL AND TOKENIZER ###
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
EOS_TOKEN = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
peft_config = LoraConfig(r=12, lora_alpha=32, lora_dropout=0.1)
model = get_peft_model(model, peft_config)
model.cuda()


### LOAD AND PRE-PROCESS DATASET ###
dataset = load_dataset(DATASET)
len_prio=len(dataset['train'])

def formatting_prompts_func(examples):
    formatted_texts = []

    for messages in examples["messages"]:  
        if not isinstance(messages, list): 
            formatted_texts.append("")
            continue

        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False  
        )

        tokenized = tokenizer(formatted_text, truncation=False, return_length=True)
        if tokenized["length"][0] < MAX_SEQ_LENGTH - 3:
            formatted_texts.append(formatted_text + EOS_TOKEN)
        else:
            formatted_texts.append(None)  # Mark for removal

    return {"text": formatted_texts}  

# Apply function to dataset
dataset = dataset.map(formatting_prompts_func, batched=True)
dataset = dataset.filter(lambda x: x["text"] is not None)  # Remove None values
len_after=len(dataset['train'])
print(f"No of instances removed: {len_prio-len_after}")

print(dataset["train"][:20])
# Shorten for demo
dataset['train'] = dataset['train'].select(range(200000))
dataset['test'] = dataset['test'].select(range(10000))
dataset['eval'] = dataset['eval'].select(range(10000))
print(f"Example from train set: {dataset['train'][0]}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          # Directory to save model and logs
    eval_strategy="epoch",    # Evaluate at the end of each epoch
    save_strategy="epoch",          # Save the model after every epoch
    logging_steps=1,              # Log every 100 steps
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,   # Batch size for evaluation
    num_train_epochs=1,             # Total number of training epochs
    learning_rate=2e-4,             # Learning rate
    warmup_steps=500,               # Number of warmup steps for the scheduler
    weight_decay=0.01,              # Strength of weight decay
    save_total_limit=2,             # Limit the total number of saved checkpoints
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    args=training_args,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_text_field="text"
)

trainer.train()

trainer.save_model(f"{OUTPUT_DIR}/final")