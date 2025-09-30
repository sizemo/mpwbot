from datasets import load_dataset, DatasetDict

# Step 1: Load your dataset
train_data_paths = ['/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00000-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00001-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00002-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00003-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00004-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00005-of-00006.arrow']
test_data_path = ['/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/test/data-00000-of-00002.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/test/data-00001-of-00002.arrow']

def load_dataset_from_arrow(paths):
    return load_dataset('arrow', data_files=paths)

train_dataset = load_dataset_from_arrow(train_data_paths)
eval_dataset = load_dataset_from_arrow(test_data_path)

# Ensure the datasets are in the correct format
train_dataset = train_dataset['train']
eval_dataset = eval_dataset['train']

# Step 2: Count tokens in each sample and sum up the total number of tokens
def count_tokens(dataset):
    token_count = 0
    for item in dataset:
        if 'input_ids' in item:
            token_count += len(item['input_ids'])  # Assuming your data is stored under 'input_ids'
        else:
            print(f"Warning: No 'input_ids' key found in item: {item}")
    return token_count


train_token_count = count_tokens(train_dataset)
eval_token_count = count_tokens(eval_dataset)

print(f"Total number of tokens in training dataset: {train_token_count}")
print(f"Total number of tokens in evaluation dataset: {eval_token_count}")