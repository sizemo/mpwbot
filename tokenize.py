from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

transcripts = '/Users/alexandra/Documents/Programming/mpwbot/cleaned_transcripts.txt'
model_name = 'gpt2'  # Replace with your chosen model
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
     return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = Dataset.from_dict({'text': [transcripts]}).map(tokenize_function, batched=True)
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)