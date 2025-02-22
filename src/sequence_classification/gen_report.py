import torch
from transformers import AutoTokenizer, LlamaForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODEL_ID = "turkishnlp/llama3.2-1b-toxicity-binary-class"
MODEL_ID = "turkishnlp/llama3.2-1b-toxicity-class" 
DATASET_PATH = "turkishnlp/toxicity_detection_labeled"
MD_REPORT_PATH = f"./eval_results/{MODEL_ID.split('/')[-1]}_report.md"

id2label = { "0": "OTHER", "1": "PROFANITY", "2": "RACIST", "3": "INSULT", "4": "SEXIST" }
label2id = {v: k for k, v in id2label.items()}

print("Loading dataset...")
dataset = load_dataset(DATASET_PATH)
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['text'])
tokenized_datasets.set_format(type="torch", device=device)

print("Loading model...")
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_ID,
    id2label=id2label,
    label2id=label2id,
)
model.config.pad_token_id = tokenizer.pad_token_id
model = model.to(device)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

training_args = TrainingArguments(
    per_device_eval_batch_size=128,
    output_dir="./results",
    logging_dir="./logs",
    report_to="none",
)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Running evaluation...")
eval_results = trainer.evaluate()

predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
predicted_labels = np.argmax(predictions, axis=1)
cm = confusion_matrix(labels, predicted_labels)

def summarize_dataset(dataset):
    summary = {
        "train_size": len(dataset["train"]),
        "test_size": len(dataset["test"]),
    }
    return summary

dataset_summary = summarize_dataset(dataset)

def format_dataset_summary(summary):
    return f"""
## Dataset Summary
- **Training Set Size:** {summary["train_size"]}
- **Test Set Size:** {summary["test_size"]}
"""

def format_confusion_matrix(cm, labels):
    header = "| " + " | ".join([""] + labels) + " |\n"
    separator = "| " + " | ".join(["---"] * (len(labels) + 1)) + " |\n"
    rows = []
    for label, row in zip(labels, cm):
        rows.append("| " + " | ".join([label] + list(map(str, row))) + " |\n")
    return header + separator + "".join(rows)

print("Generating Markdown report...")
labels = list(id2label.values())
conf_matrix_md = format_confusion_matrix(cm, labels)

dataset_summary_md = format_dataset_summary(dataset_summary)
eval_results_md = f"""
## Evaluation Metrics
- **Accuracy:** {eval_results["eval_accuracy"]:.4f}
- **Precision:** {eval_results["eval_precision"]:.4f}
- **Recall:** {eval_results["eval_recall"]:.4f}
- **F1-Score:** {eval_results["eval_f1"]:.4f}
"""

md_report = f"""
# Sequence Classification Model Evaluation Report

{eval_results_md}

## Confusion Matrix
{conf_matrix_md}

{dataset_summary_md}
"""

# Save report to a file
with open(MD_REPORT_PATH, "w") as f:
    f.write(md_report)

print(f"Report saved to {MD_REPORT_PATH}")
