from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset

MODEL_ID = "meta-llama/Llama-3.2-1B"
DATASET_NAME = "musabg/wikipedia-oscar-tr" 
OUTPUT_DIR = "../../artifacts/models/llama3.2-1b_cpt"
REPO_NAME = 'turkishnlp/oscar-wilde'
MAX_SEQ_LENGTH = 1024

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.cuda()

dataset = load_dataset(DATASET_NAME).train_test_split(test_size=0.1) 

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=MAX_SEQ_LENGTH)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,      
    evaluation_strategy="epoch",  
    save_strategy="epoch",        
    per_device_train_batch_size=2, 
    per_device_eval_batch_size=2,
    num_train_epochs=2,           
    learning_rate=5e-5,           
    warmup_steps=1000,            
    weight_decay=0.01,            
    save_total_limit=1,           
    logging_steps=100,            
    fp16=True,                    
    report_to="none",             
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

### SAVE FINAL MODEL ###
trainer.save_model(f"{OUTPUT_DIR}/final")

try:
    model.push_to_hub(REPO_NAME)
except Exception as e:
    breakpoint()