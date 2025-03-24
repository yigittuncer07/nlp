from sklearn.metrics import precision_recall_fscore_support

def normalize(text):
    return text.lower().strip()

def has_overlap(pred_span, true_span):
    pred_tokens = set(normalize(pred_span).split())
    true_tokens = set(normalize(true_span).split())
    return len(pred_tokens & true_tokens) > 0

def evaluate_overlap_based(results):
    y_true = []
    y_pred = []

    for entry in results:
        true_entities = entry['true']
        pred_entities = entry['pred']

        matched = set()
        for pred_text, pred_label in pred_entities:
            found = False
            for i, (true_text, true_label) in enumerate(true_entities):
                if true_label == pred_label and has_overlap(pred_text, true_text) and i not in matched:
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                    matched.add(i)
                    found = True
                    break
            if not found:
                y_true.append("None")
                y_pred.append(pred_label)

        for i, (true_text, true_label) in enumerate(true_entities):
            if i not in matched:
                y_true.append(true_label)
                y_pred.append("None")

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(y_true)
    }

import json
with open("results.json", encoding="utf-8") as f:
    results = json.load(f)

metrics = evaluate_overlap_based(results)
print(metrics)
