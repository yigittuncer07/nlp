from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "../../artifacts/models/meta_unllama_ner/final"
MODEL_REPO_NAME = "turkishnlp/UNllama3.2-1b-instruct-ner" 

def push_to_hub(model_dir, repo_name):
    print("Loading model and tokenizer...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Pushing model to Hugging Face Hub at {repo_name}...")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model successfully pushed to Hugging Face Hub: {repo_name}")

if __name__ == "__main__":
    push_to_hub(MODEL_DIR, MODEL_REPO_NAME)
