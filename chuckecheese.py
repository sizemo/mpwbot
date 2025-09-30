import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, DatasetDict

transcripts_path = '/Users/alexandra/Documents/Programming/mpwbot/cleaned_transcripts.txt'
model_name = 'gpt2'  # Replace with your chosen model

# Load tokenizer and add padding token if it doesn't exist
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Read transcripts from file
with open(transcripts_path, 'r', encoding='utf-8') as file:
    texts = [line.strip() for line in file if line.strip()]

# Ensure texts is a list of strings
assert isinstance(texts, list), "texts should be a list of strings"

# Create dataset and tokenize
dataset = Dataset.from_dict({'text': texts})
tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)

# Save the split datasets to a new directory (optional)
output_dir = 'tokenized_datasets'
os.makedirs(output_dir, exist_ok=True)
train_test_split.save_to_disk(output_dir)

print(train_test_split)
