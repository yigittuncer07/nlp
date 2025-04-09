import torch
from transformers import AutoTokenizer, LlamaForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "turkishnlp/llama3.2-1b-instruct-ner"
DATASET_NAME="huggingface_dataset_path"
MD_REPORT_PATH = f"./eval_results/{MODEL_PATH.split('/')[-1]}_report.md"

print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)
id2label = { 0: "B-LOCATION", 1: "B-ORGANIZATION", 2: "B-PERSON", 3: "I-LOCATION", 4: "I-ORGANIZATION", 5: "I-PERSON", 6: "O" }
label2id = {v: k for k, v in id2label.items()}

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForTokenClassification.from_pretrained(MODEL_PATH, num_labels=len(label2id), id2label=id2label, label2id=label2id)
model = model.to(device)

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
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

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

    conf_matrix = confusion_matrix(true_labels, true_predictions, labels=range(len(id2label)))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "confusion_matrix": conf_matrix,
    }

def summarize_dataset(dataset, label_to_id):
    summary = {
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]),
        "test_size": len(dataset["test"]),
    }

    label_counts = {label: 0 for label in label_to_id.keys()}
    for example in dataset["train"]:
        for tag in example["tags"]:
            label_counts[tag] += 1

    summary["label_distribution"] = {
        label: count for label, count in sorted(label_counts.items(), key=lambda x: label_to_id[x[0]])
    }

    input_lengths = [len(example["tokens"]) for example in dataset["train"]]
    summary["average_input_length"] = sum(input_lengths) / len(input_lengths)

    return summary

def format_dataset_summary(summary):
    label_dist_md = "\n".join(
        [f"- **{label}**: {count}" for label, count in summary["label_distribution"].items()]
    )
    return f"""
## Dataset Summary
- **Training Set Size:** {summary["train_size"]}
- **Validation Set Size:** {summary["validation_size"]}
- **Test Set Size:** {summary["test_size"]}
- **Average Input Length:** {summary["average_input_length"]:.2f}

### Label Distribution (Training Set)
{label_dist_md}
"""

def format_confusion_matrix(cm, labels):
    header = "| " + " | ".join([""] + labels) + " |\n"
    separator = "| " + " | ".join(["---"] * (len(labels) + 1)) + " |\n"
    rows = []
    for label, row in zip(labels, cm):
        rows.append("| " + " | ".join([label] + list(map(str, row))) + " |\n")
    return header + separator + "".join(rows)

training_args = TrainingArguments(
    output_dir="./eval_results",
    per_device_eval_batch_size=64,
    logging_dir="./logs",
    report_to="none",
)

print("Initializing trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

print("Running evaluation...")
eval_results = trainer.evaluate()

print("Summarizing dataset...")
dataset_summary = summarize_dataset(dataset, label2id)
dataset_summary_md = format_dataset_summary(dataset_summary)

print("Generating Markdown report...")
labels = [id2label[i] for i in range(len(id2label))]
conf_matrix_md = format_confusion_matrix(eval_results["eval_confusion_matrix"], labels)

md_report = f"""
# NER Model Evaluation Report

## Evaluation Metrics
- **Precision:** {eval_results["eval_precision"]:.4f}
- **Recall:** {eval_results["eval_recall"]:.4f}
- **F1-Score:** {eval_results["eval_f1"]:.4f}
- **Accuracy:** {eval_results["eval_accuracy"]:.4f}

## Confusion Matrix
{conf_matrix_md}

{dataset_summary_md}
"""

with open(MD_REPORT_PATH, "w") as f:
    f.write(md_report)

print(f"Report saved to {MD_REPORT_PATH}")
