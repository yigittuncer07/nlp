from huggingface_hub import InferenceClient
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import json
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logging.basicConfig(
    filename='ner_inference-3b.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PROMPT = "Aşağıdaki kelime listesi üzerinde Adlandırılmış Varlık Tanıma (NER) işlemi yap ve her bir kelimeyi şu etiketlerden biriyle sınıflandır: B-LOC (konumun başlangıcı), I-LOC (konumun içi), B-ORG (kuruluşun başlangıcı), I-ORG (kuruluşun içi), B-PER (kişinin başlangıcı), I-PER (kişinin içi), O (diğer).\n\nKelime Listesi: '{}'\n\nSonucu 'entities' adlı bir dizi içeren bir JSON nesnesi olarak döndür. Her varlık 'kelime' ve 'etiket' özelliklerine sahip olmalıdır."
DATASET_NAME = "turkishnlp/WikiANN-Turkish-JSON-Format"

dataset = load_dataset(DATASET_NAME)
dataset = dataset['validation']

client = InferenceClient("http://localhost:1411")

ner_schema = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kelime": {"type": "string"},
                    "etiket": {"type": "string", "enum": ["B-LOC", "B-ORG", "B-PER", "I-LOC", "I-ORG", "I-PER", "O"]}
                },
                "required": ["kelime", "etiket"]
            }
        }
    },
    "required": ["entities"]
}

def ner_infer(words):
    try:
        resp = client.text_generation(
            PROMPT.format(words),
            max_new_tokens=512,
            seed=42,
            grammar={"type": "json", "value": ner_schema},
        )
        return resp
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return None

def evaluate_ner(dataset, num_samples=None):
    if num_samples:
        subset = dataset[:num_samples]
    else:
        subset = dataset
        
    tokens_list = subset["tokens"]
    tags_list = subset["tags"]
    
    all_true_tags = []
    all_pred_tags = []
    broken_responses_count = 0
    
    for idx, (tokens, tags) in enumerate(zip(tokens_list, tags_list)):
        logging.info(f"Processing sample {idx+1}/{len(tokens_list)}")
        
        response = ner_infer(str(tokens))
        
        logging.info(f"Sample {idx+1} - Raw response: {response}")
        
        try:
            result = json.loads(response)
            entities = result.get("entities", [])
            
            if len(entities) != len(tokens):
                logging.warning(f"Sample {idx+1} - Length mismatch: entities({len(entities)}) != tokens({len(tokens)})")
                broken_responses_count += 1
                continue
                
            pred_tags = [entity["etiket"] for entity in entities]
            
            all_true_tags.extend(tags)
            all_pred_tags.extend(pred_tags)
            
        except json.JSONDecodeError:
            logging.error(f"Sample {idx+1} - Invalid JSON response")
            broken_responses_count += 1
            continue
        except Exception as e:
            logging.error(f"Sample {idx+1} - Error processing response: {e}")
            broken_responses_count += 1
            continue
    
    if len(all_true_tags) > 0 and len(all_pred_tags) > 0:
        tags_set = sorted(list(set(all_true_tags + all_pred_tags)))
        tag_to_idx = {tag: idx for idx, tag in enumerate(tags_set)}
        
        true_indices = [tag_to_idx[tag] for tag in all_true_tags]
        pred_indices = [tag_to_idx[tag] for tag in all_pred_tags]
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_indices, pred_indices, average='weighted'
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_indices, pred_indices, average='macro'
        )
        accuracy = accuracy_score(true_indices, pred_indices)
        conf_matrix = confusion_matrix(true_indices, pred_indices)
        
        results = {
            "total_samples": len(tokens_list),
            "broken_responses": broken_responses_count,
            "valid_samples": len(tokens_list) - broken_responses_count,
            "metrics": {
                "accuracy": float(accuracy),
                "weighted": {
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1)
                },
                "macro": {
                    "precision": float(macro_precision),
                    "recall": float(macro_recall),
                    "f1": float(macro_f1)
                }
            },
            "confusion_matrix": conf_matrix.tolist(),
            "labels": tags_set
        }
        
        return results
    else:
        logging.error("No valid predictions collected, cannot calculate metrics")
        return None

def save_results_to_markdown(results):
    if results is None:
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ner_evaluation_results_{timestamp}.md"
    
    with open(filename, "w") as f:
        f.write("# Named Entity Recognition Evaluation Results\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Information\n")
        f.write(f"- Dataset: {DATASET_NAME}\n")
        f.write(f"- Total samples evaluated: {results['total_samples']}\n")
        f.write(f"- Broken responses: {results['broken_responses']} ({results['broken_responses']/results['total_samples']*100:.2f}%)\n")
        f.write(f"- Valid samples: {results['valid_samples']} ({results['valid_samples']/results['total_samples']*100:.2f}%)\n\n")
        
        f.write("## Performance Metrics\n")
        f.write(f"- Accuracy: {results['metrics']['accuracy']:.4f}\n\n")
        
        f.write("### Weighted Metrics\n")
        f.write(f"- Precision: {results['metrics']['weighted']['precision']:.4f}\n")
        f.write(f"- Recall: {results['metrics']['weighted']['recall']:.4f}\n")
        f.write(f"- F1 Score: {results['metrics']['weighted']['f1']:.4f}\n\n")
        
        f.write("### Macro Metrics\n")
        f.write(f"- Precision: {results['metrics']['macro']['precision']:.4f}\n")
        f.write(f"- Recall: {results['metrics']['macro']['recall']:.4f}\n")
        f.write(f"- F1 Score: {results['metrics']['macro']['f1']:.4f}\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write("```\n")
        f.write(f"Labels: {results['labels']}\n\n")
        
        conf_matrix = np.array(results['confusion_matrix'])
        formatted_matrix = ""
        for row in conf_matrix:
            formatted_matrix += " ".join([f"{val:4d}" for val in row]) + "\n"
        f.write(formatted_matrix)
        f.write("```\n\n")
        
        # Save visualization of confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=results['labels'], yticklabels=results['labels'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.tight_layout()
        
        # Save confusion matrix plot
        cm_filename = f"confusion_matrix_{timestamp}.png"
        plt.savefig(cm_filename)
        f.write(f"![Confusion Matrix]({cm_filename})\n")
        
    print(f"Results saved to {filename}")

# Main execution
if __name__ == "__main__":
    num_samples = 5000  # set to none for full eval
    
    print(f"Starting NER evaluation on {DATASET_NAME}...")
    results = evaluate_ner(dataset, num_samples)
    
    if results:
        save_results_to_markdown(results)
        print(f"Evaluation completed. Total samples: {results['total_samples']}, Valid: {results['valid_samples']}, Broken: {results['broken_responses']}")
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}, F1-score: {results['metrics']['weighted']['f1']:.4f}")
    else:
        print("Evaluation failed due to insufficient valid responses.")