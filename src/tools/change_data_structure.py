from datasets import load_dataset

dataset = load_dataset("turkishnlp/turkish_summarization")
from tqdm import tqdm

train = dataset['train']
test = dataset['test']
eval = dataset['eval']

new_train = []
new_test = []
new_eval = []
system_prompt = ""
for entry in tqdm(train):
    new_train.append({
        "messages":[
            {"role": "system", "content": "Sen bir metin özetleme asistanısın. Sana verilen metinleri analiz et ve önemli noktaları içeren kısa özetler yaz"},
            {"role": "user", "content": entry.get("input")},
            {"role": "assistant", "content": entry.get("output")}]})

for entry in test:
    new_test.append({
        "messages":[
            {"role": "system", "content": "Sen bir metin özetleme asistanısın. Sana verilen metinleri analiz et ve önemli noktaları içeren kısa özetler yaz"},
            {"role": "user", "content": entry.get("input")},
            {"role": "assistant", "content": entry.get("output")}]})

for entry in eval:
    new_eval.append({
        "messages":[
            {"role": "system", "content": "Sen bir metin özetleme asistanısın. Sana verilen metinleri analiz et ve önemli noktaları içeren kısa özetler yaz"},
            {"role": "user", "content": entry.get("input")},
            {"role": "assistant", "content": entry.get("output")}]})
    
from datasets import Dataset, DatasetDict, load_dataset

# Convert your new_train, new_test, and new_eval lists to Dataset objects
new_train_dataset = Dataset.from_list(new_train)
new_test_dataset = Dataset.from_list(new_test)
new_eval_dataset = Dataset.from_list(new_eval)

# Combine into a DatasetDict
new_dataset = DatasetDict({
    "train": new_train_dataset,
    "test": new_test_dataset,
    "eval": new_eval_dataset
})

new_dataset.save_to_disk("artifacts/datasets/turkish_summarization")
# Push the dataset to the Hugging Face Hub
# new_dataset.push_to_hub("turkishnlp/conversation_summarization")

    
