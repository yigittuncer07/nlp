from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from modeling_llama import UnmaskingLlamaForTokenClassification

# MODEL_ID="meta-llama/Llama-3.2-1b-Instruct"
MODEL_ID = 'turkishnlp/Llama-3.2-1B-Instruct-CPT-oscar-Unsloth'
DATASET_NAME="turkishnlp/WikiANN-Turkish-JSON-Format"
SAVE_DIR = "../../artifacts/models/unsloth/cpt-meta_unllama_ner_wikiann"

id2label = { 0: "B-LOC", 1: "B-ORG", 2: "B-PER", 3: "I-LOC", 4: "I-ORG", 5: "I-PER", 6: "O" }

label2id = {v: k for k, v in id2label.items()}

# Load the dataset
dataset = load_dataset(DATASET_NAME)

model = UnmaskingLlamaForTokenClassification.from_pretrained(MODEL_ID, num_labels=len(label2id), id2label=id2label, label2id=label2id)
model.to("cuda")

peft_config = LoraConfig(task_type=TaskType.TOKEN_CLS, inference_mode=False, r=12, lora_alpha=32, lora_dropout=0.1, )
model = get_peft_model(model, peft_config)
model.to("cuda")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:  # Padding
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # New token
                label_ids.append(label2id[label[word_idx]])
            else:  # Same token
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    true_labels = labels.flatten()
    true_predictions = predictions.flatten()

    valid_indices = true_labels != -100
    true_labels = true_labels[valid_indices]
    true_predictions = true_predictions[valid_indices]

    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, true_predictions, average="weighted"
    )
    accuracy = accuracy_score(true_labels, true_predictions)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }

tokenized_datasets = dataset.map(tokenize_and_align_labels,batched=True)

training_args = TrainingArguments(
    output_dir=SAVE_DIR,  
    evaluation_strategy="epoch",  
    learning_rate=1e-4,
    per_device_train_batch_size=32,  
    per_device_eval_batch_size=32,  
    num_train_epochs=1,  
    weight_decay=0.01, 
    save_strategy="epoch", 
    logging_steps=1, 
    save_total_limit=1,  
    report_to="none", 
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=tokenized_datasets["train"], 
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer, 
    compute_metrics=compute_metrics,  
)

trainer.train()
