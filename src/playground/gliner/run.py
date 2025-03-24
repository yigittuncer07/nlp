from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from gliner import GLiNER
from datasets import load_dataset
from tqdm import tqdm

def evaluate_gliner(model, dataset, labels):
    all_true_tags = []
    all_pred_tags = []
    
    comparison = []

    label_map = {
        "ORG": "Organization",
        "PER": "Person",
        "LOC": "Location"
    }

    for sample in tqdm(dataset['validation']):
        tokens = sample['tokens']
        tags = sample['tags']

        merged = []
        i = 0
        while i < len(tokens):
            tag = tags[i]
            token = tokens[i]
            if tag == 'O':
                i += 1
                continue
            if tag.startswith('B-'):
                ent_type = tag[2:]
                entity_tokens = [token]
                i += 1
                while i < len(tokens) and tags[i] == f'I-{ent_type}':
                    entity_tokens.append(tokens[i])
                    i += 1
                readable_type = label_map.get(ent_type, ent_type)
                merged.append((' '.join(entity_tokens), readable_type))
            else:
                i += 1

        text = ' '.join(tokens)
        predicted_entities = model.predict_entities(text, labels, threshold=0.5)
        predictions = [(item.get('text'), item.get('label')) for item in predicted_entities]
        
        comparison.append({'text': text, 'true': merged, 'pred': predictions})

        # all_true_tags.extend(true_tags)
        # all_pred_tags.extend(pred_tags)
    
    # # 4. Calculate metrics
    # # Convert tags to indices for confusion matrix
    # unique_tags = sorted(list(set(all_true_tags + all_pred_tags)))
    # tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
    
    # true_indices = [tag_to_idx[tag] for tag in all_true_tags]
    # pred_indices = [tag_to_idx[tag] for tag in all_pred_tags]
    
    # # Calculate metrics
    # precision, recall, f1, _ = precision_recall_fscore_support(all_true_tags, all_pred_tags, average='macro')
    # accuracy = accuracy_score(all_true_tags, all_pred_tags)
    
    # cm = confusion_matrix(true_indices, pred_indices)
    
    # results = {
    #     'accuracy': accuracy,
    #     'precision': precision,
    #     'recall': recall,
    #     'f1': f1,
    #     'confusion_matrix': cm,
    #     'tag_mapping': unique_tags
    # }
    
    # return results
    return comparison

labels = ["Location", "Organization", "Person"]
model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5")
model.to('cuda')
dataset = load_dataset("turkishnlp/WikiANN-Turkish-JSON-Format")

eval_results = evaluate_gliner(model, dataset, labels)

import json
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(eval_results, f, ensure_ascii=False, indent=4)


# # Print results
# print(f"Accuracy: {eval_results['accuracy']:.4f}")
# print(f"Precision: {eval_results['precision']:.4f}")
# print(f"Recall: {eval_results['recall']:.4f}")
# print(f"F1 Score: {eval_results['f1']:.4f}")