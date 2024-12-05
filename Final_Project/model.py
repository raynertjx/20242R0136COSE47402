from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import os
import torch

def get_latest_checkpoint(model_dir):
    # List all subdirectories in the model directory
    checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
    # Sort by checkpoint number (numerical order)
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('-')[-1]))[-1]
    return os.path.join(model_dir, latest_checkpoint)

# Step 1: Load and Preprocess RecipeNLG Dataset
def load_recipenlg():
    print("Loading RecipeNLG dataset...")
    # Load the dataset from drive
    # Dataset was downloaded from https://recipenlg.cs.put.poznan.pl/
    path = "/content/drive/MyDrive/Colab Notebooks/"
    dataset = load_dataset("recipe_nlg", data_dir=path)

    # Preprocess: Combine title, ingredients, and directions into a single text input
    def preprocess_data(example):
        return {
            "text": f"Title: {example['title']} Ingredients: {example['ingredients']} Instructions: {example['directions']}"
        }

    dataset = dataset.map(preprocess_data, remove_columns=["title", "ingredients", "directions"])

    # Ensure exact sizes for training and testing
    total_size = len(dataset["train"])
    train_size = 8000
    test_size = 2000

    dataset_split = dataset["train"].train_test_split(
        test_size=test_size / total_size,  # Calculate test split ratio
        seed=42
    )

    dataset = DatasetDict({
        "train": dataset_split["train"].select(range(train_size)),  # Limit train split to 8000
        "test": dataset_split["test"].select(range(test_size)),    # Limit test split to 2000
    })

    print(f"Dataset loaded and reduced: {dataset}")
    return dataset

# Step 2: Fine-Tune Pre-Trained Model
def fine_tune_model(dataset, model_name="gpt2", output_dir="fine_tuned_model"):
    print(f"Fine-tuning {model_name} on RecipeNLG...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize the dataset
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )
        tokenized["labels"] = tokenized["input_ids"].copy()  # Add labels for loss computation
        return tokenized

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,  # Moderate batch size
        num_train_epochs=3,  # Use only 1 epoch for initial testing
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        weight_decay=0.01,
        push_to_hub=False,
        report_to="none",
        fp16=True
    )

    # Trainer for fine-tuning
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    trainer.train()
    print(f"Model fine-tuned and saved to {output_dir}.")
    return output_dir

# Step 3: Generate Recipes
def generate_recipe(model_path, prompt, max_length=200):
    print("Loading fine-tuned model for generation...")
    latest_checkpoint = get_latest_checkpoint(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(latest_checkpoint)
    model = GPT2LMHeadModel.from_pretrained(latest_checkpoint)
    tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.8,
        repetition_penalty=1.5,  
        top_k=50,  
        top_p=0.95, 
    )
    # Decode and split by line
    recipe = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return recipe

# Step 4: Main Function
if __name__ == "__main__":
    # Load RecipeNLG dataset
    recipe_nlg_dataset = load_recipenlg()

    # Fine-tune the model
    fine_tuned_model_dir = fine_tune_model(recipe_nlg_dataset)
    print(fine_tuned_model_dir)

    # Generate some sample recipes
    user_prompt1 = "Title: Chicken and Broccoli Rice Bowl, Ingredients: rice, chicken, broccoli, soy sauce, garlic, onion, olive oil"
    user_prompt2 = "Title: Aglio Olio, Ingredients: pasta, olive oil, garlic, chili"
    user_prompt3 = "Title: Caesar Salad, Ingredients: olive oil, lettuce, tomato, onion, cheese"
    generated_recipe1 = generate_recipe(fine_tuned_model_dir, user_prompt1)
    generated_recipe2 = generate_recipe(fine_tuned_model_dir, user_prompt2)
    generated_recipe3 = generate_recipe(fine_tuned_model_dir, user_prompt3)
    print("\nGenerated Recipe 1:\n")
    print(generated_recipe1)y
    print("\nGenerated Recipe 3:\n")
    print(generated_recipe3)