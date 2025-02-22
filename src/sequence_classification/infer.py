from transformers import AutoTokenizer, LlamaForSequenceClassification
import torch

# Constants
MODEL_PATH = '../../artifacts/models/toxicity_classifier/binary_toxicity_classifier'
id2label={0: 'OTHER', 1: 'TOXIC'}
label2id = {v: k for k, v in id2label.items()}

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Load the model and tokenizer
model = LlamaForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()  # Set to evaluation mode
model.cuda()  # Move model to GPU


def classify_sentence(sentence):
    """Tokenize the input and return prediction percentages."""
    inputs = tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=256).to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1).squeeze().tolist()
    return {id2label[i]: prob for i, prob in enumerate(probabilities)}

# Interactive loop
print("Enter a sentence to classify its toxicity (enter 'q' to quit):")
while True:
    user_input = input("Sentence: ").strip()
    if user_input.lower() == 'q':
        print("Exiting...")
        break
    prediction_percentages = classify_sentence(user_input)
    print("Prediction Percentages:")
    for label, percentage in prediction_percentages.items():
        print(f"  {label}: {percentage * 100:.2f}%")
    print()
