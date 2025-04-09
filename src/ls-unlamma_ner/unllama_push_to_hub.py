from transformers import AutoTokenizer
from modeling_llama import UnmaskingLlamaForTokenClassification


MODEL_DIR = "/home/yigittuncer/nlp/src/unlamma_ner/models/unsloth/cpt-meta_unllama_ner_wikiann/final"
MODEL_REPO_NAME = "turkishnlp/TR-UNllama3.2-1b-instruct-ner-wikiann" 
id2label = { 0: "B-LOC", 1: "B-ORG", 2: "B-PER", 3: "I-LOC", 4: "I-ORG", 5: "I-PER", 6: "O" }

label2id = {v: k for k, v in id2label.items()}
def push_to_hub(model_dir, repo_name):
    print("Loading model and tokenizer...")
    model = UnmaskingLlamaForTokenClassification.from_pretrained(model_dir, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Pushing model to Hugging Face Hub at {repo_name}...")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model successfully pushed to Hugging Face Hub: {repo_name}")

if __name__ == "__main__":
    push_to_hub(MODEL_DIR, MODEL_REPO_NAME)
 