import torch
from transformers import pipeline
from datasets import load_dataset
import json
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import numpy as np
from pathlib import Path

DATASET = "turkishnlp/conversation_summarization"

model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "turkishnlp/llama3.2-1b-sum-finetune-final"

eval_file_name = f"inferences/{model_id.split('/')[-1]}_mass_inference.json"
markdown_file_name = f"evaluations/{model_id.split('/')[-1]}_evaluation.md"

# Load the dataset
print("Loading dataset...")
dataset = load_dataset(DATASET)

def compute_token_stats(dataset_split):
    word_counts = []
    for item in tqdm(dataset_split):
        text = item.get('messages')[1].get('content')
        word_counts.append(len(text.split()))
    return {
        "instances": len(dataset_split),
        "avg_word_count": np.mean(word_counts),
    }

train_stats = compute_token_stats(dataset["train"])
test_stats = compute_token_stats(dataset["test"])

# Load the mass inference file
print("Loading mass inference results...")
with open(eval_file_name, "r", encoding="utf-8") as f:
    results = json.load(f)

# Extract expected and generated responses
expected_responses = [item["expected_response"] for item in results]
generated_responses = [item["generated_response"] for item in results]

# Calculate BERT scores
print("Calculating BERT scores...")
P, R, F1 = score(generated_responses, expected_responses, lang="tr", verbose=True)

# Compute average BERT scores
avg_precision = torch.mean(P).item()
avg_recall = torch.mean(R).item()
avg_f1 = torch.mean(F1).item()

# Calculate ROUGE scores
print("Calculating ROUGE scores...")
rouge_scorer_instance = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
for ref, gen in zip(expected_responses, generated_responses):
    scores = rouge_scorer_instance.score(ref, gen)
    rouge1_scores.append(scores["rouge1"].fmeasure)
    rouge2_scores.append(scores["rouge2"].fmeasure)
    rougeL_scores.append(scores["rougeL"].fmeasure)

avg_rouge1 = np.mean(rouge1_scores)
avg_rouge2 = np.mean(rouge2_scores)
avg_rougeL = np.mean(rougeL_scores)

# Calculate BLEU scores
print("Calculating BLEU scores...")
bleu_score = corpus_bleu([[ref.split()] for ref in expected_responses], [gen.split() for gen in generated_responses])

# Generate markdown file
print("Saving results to markdown file...")
markdown_content = f"""
# Evaluation Results for {model_id}

## Dataset Information
- **Dataset Name**: {DATASET}
- **Train Instances**: {train_stats['instances']}
- **Average Words per Train Instance**: {train_stats['avg_word_count']:.2f}
- **Test Instances**: {test_stats['instances']}
- **Average Words per Test Instance**: {test_stats['avg_word_count']:.2f}

## Evaluation Metrics
### BERT Scores
- **Average Precision**: {avg_precision:.4f}
- **Average Recall**: {avg_recall:.4f}
- **Average F1 Score**: {avg_f1:.4f}

### ROUGE Scores
- **Average ROUGE-1**: {avg_rouge1:.4f}
- **Average ROUGE-2**: {avg_rouge2:.4f}
- **Average ROUGE-L**: {avg_rougeL:.4f}

### BLEU Score
- **Average BLEU**: {bleu_score:.4f}

## Notes
- This evaluation is performed on the {DATASET} dataset.
- Model evaluated: {model_id}
"""

Path(markdown_file_name).write_text(markdown_content, encoding="utf-8")
print(f"Results saved to {markdown_file_name}")
