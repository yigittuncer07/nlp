from datasets import load_dataset
from transformers import LlamaForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from peft import get_peft_model, LoraConfig, TaskType

# Constants
MODEL_ID = 'meta-llama/Llama-3.2-1B-Instruct'
DATASET_PATH = 'turkishnlp/toxicity_detection_labeled'
OUTPUT_DIR = '../../artifacts/models/toxicity_classifier'

id2label = { "0": "OTHER", "1": "PROFANITY", "2": "RACIST", "3": "INSULT", "4": "SEXIST" } 
label2id ={ "OTHER": 0, "PROFANITY": 1, "RACIST": 2, "INSULT": 3, "SEXIST": 4 }

dataset = load_dataset(DATASET_PATH)

# Check the splits
print(f"Dataset splits: {dataset.keys()}")
print(f"Example from train set: {dataset['train'][0]}")

# Load Model and Tokenizer
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id,
)
peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1)

model = get_peft_model(model, peft_config)
model.cuda()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize Dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format(type="torch", device="cuda")

# Define DataLoaders
train_loader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=8)
test_loader = DataLoader(tokenized_datasets['test'], batch_size=8)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,          # Directory to save model and logs
    evaluation_strategy="epoch",    # Evaluate at the end of each epoch
    save_strategy="epoch",          # Save the model after every epoch
    logging_dir="./logs",           # Directory for logs
    logging_steps=1,              # Log every 100 steps
    per_device_train_batch_size=8,  # Batch size for training
    per_device_eval_batch_size=8,   # Batch size for evaluation
    num_train_epochs=1,             # Total number of training epochs
    learning_rate=2e-4,             # Learning rate
    warmup_steps=500,               # Number of warmup steps for the scheduler
    weight_decay=0.01,              # Strength of weight decay
    save_total_limit=2,             # Limit the total number of saved checkpoints
    load_best_model_at_end=True,    # Load the best model at the end of training
    metric_for_best_model="accuracy",  # Metric to determine the best model
    greater_is_better=True,         # Whether a higher metric value is better
)


# Define a compute metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# Init trainer
trainer = Trainer(
    model=model,                       # The model to train
    args=training_args,                # Training arguments
    train_dataset=tokenized_datasets['train'],  # Training dataset
    eval_dataset=tokenized_datasets['test'],  # Test dataset
    tokenizer=tokenizer,               # Tokenizer
    compute_metrics=compute_metrics    # Evaluation metrics
)

# Fine-tune the model
trainer.train()

# Evaluate on the test dataset
results = trainer.evaluate(tokenized_datasets['test'])
print("Test Results:", results)

trainer.save_model(f"{OUTPUT_DIR}/multiclass_toxicity_classifier")