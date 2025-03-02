from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "../../artifacts/models/llama3.2-1b-instruct_cpt-FULL/final"
MODEL_REPO_NAME = "turkishnlp/Llama-3.2-1B-Instruct-CPT-oscar-Unsloth-FULL" 

def push_to_hub(model_dir, repo_name):
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Pushing model to Hugging Face Hub at {repo_name}...")
    model.push_to_hub(repo_name)
    tokenizer.push_to_hub(repo_name)

    print(f"Model successfully pushed to Hugging Face Hub: {repo_name}")

if __name__ == "__main__":
    push_to_hub(MODEL_DIR, MODEL_REPO_NAME)
