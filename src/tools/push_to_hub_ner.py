from transformers import LlamaForTokenClassification, AutoTokenizer

MODEL_DIR = "../../artifacts/models/meta_llama_ner/final"
MODEL_REPO_NAME = "turkishnlp/llama3.2-1b-instruct-ner" 
id2label = { 0: "B-LOCATION", 1: "B-ORGANIZATION", 2: "B-PERSON", 3: "I-LOCATION", 4: "I-ORGANIZATION", 5: "I-PERSON", 6: "O" }

label2id = {v: k for k, v in id2label.items()}
def push_to_hub(model_dir, repo_name):
    print("Loading model and tokenizer...")
    model = LlamaForTokenClassification.from_pretrained(model_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Pushing model to Hugging Face Hub at {repo_name}...")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model successfully pushed to Hugging Face Hub: {repo_name}")

if __name__ == "__main__":
    push_to_hub(MODEL_DIR, MODEL_REPO_NAME)
