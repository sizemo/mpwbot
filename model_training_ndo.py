from transformers import GPT2LMHeadModel, GPT2TokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

# Step 1: Load your tokenizer and model
model_name = 'gpt2'  # You can use a different pre-trained model if you prefer
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 2: Set the paths to your tokenized datasets
train_data_paths = ['/Users/alexandra/Documents/Programming/mpwbot/tokenize_datasets/train/data-00000-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00001-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00002-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00003-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00004-of-00006.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/train/data-00005-of-00006.arrow']
test_data_path = ['/Users/alexandra/Documents/Programming/mpwbot/tokenized_atasets/test/data-00000-of-00002.arrow', 
'/Users/alexandra/Documents/Programming/mpwbot/tokenized_datasets/test/data-00001-of-00002.arrow']

# Step 3: Load and preprocess the data
def load_dataset_from_arrow(paths):
    return load_dataset('arrow', data_files=paths)

train_dataset = load_dataset_from_arrow(train_data_paths)
eval_dataset = load_dataset_from_arrow(test_data_path)

# Ensure the datasets are in the correct format for GPT-2
train_dataset = train_dataset['train']
eval_dataset = eval_dataset['train']

# Step 4: Set up the training arguments and trainer
output_dir = '/Users/alexandra/Documents/Programming/mpwbot/model'
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Step 5: Train the model with monitoring and early stopping
def train_with_early_stopping(trainer, max_eval_loss=float('inf'), 
patience=3):
    counter = 0
    for epoch in range(training_args.num_train_epochs):
        trainer.train()
        # Evaluate on validation set
        results = trainer.evaluate(eval_dataset)
        current_eval_loss = results['eval_loss']

        # Check if validation loss has improved
        if current_eval_loss < max_eval_loss:
            max_eval_loss = current_eval_loss
            counter = 0
        else:
            counter += 1

        # Log metrics to console and Weights & Biases (if initialized)
        print(f"Epoch {epoch + 1}, Eval Loss: {current_eval_loss:.4f}")
        
        # Early stopping check
        if counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

# Example usage
train_with_early_stopping(trainer)