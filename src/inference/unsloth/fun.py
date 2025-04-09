# !https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing#scrollTo=pCqnaKmlO1U9

from datasets import load_dataset

# Load the Turkish summarization dataset
dataset = load_dataset("turkishnlp/turkish_summarization", split="train")

# Compute output lengths
output_lengths = dataset.map(lambda x: {"output_length": len(x["output"].strip())})

# Calculate average output length
total_length = sum(output_lengths["output_length"])
average_length = total_length / len(output_lengths)

# Count how many outputs are empty or whitespace-only
num_empty_outputs = sum(1 for length in output_lengths["output_length"] if length == 0)

print("Average output length:", average_length)
print("Number of empty outputs:", num_empty_outputs)
print("Total samples:", len(output_lengths))
