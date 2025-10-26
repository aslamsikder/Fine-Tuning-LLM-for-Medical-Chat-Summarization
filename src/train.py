import os
import torch
import pandas as pd
from dotenv import load_dotenv
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login, HfApi
from datasets import Dataset
import argparse

# -----------------------------
# Helper: Format each sample
# -----------------------------
def format_chat_template(row):
    """Apply chat-style prompt for summarization."""
    messages = [
        {"role": "user", "content": f"Summarize the following medical dialogue into a SOAP note:\n\n{row['dialogue']}"},
        {"role": "model", "content": f"{row['soap']}"}
    ]

    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<bos><start_of_turn>user\n' + message['content'] + '<end_of_turn>\n' }}"
        "{% elif message['role'] == 'model' %}"
        "{{ '<start_of_turn>model\n' + message['content'] + '<end_of_turn>' }}"
        "{% endif %}"
        "{% endfor %}"
    )

    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

# -----------------------------
# Main Training Function
# -----------------------------
def main(args):
    global tokenizer

    # Load Environment Variables
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

    # Hugging Face Login
    if hf_token:
        login(token=hf_token)
        print("Logged in to Hugging Face successfully!")
    else:
        print("HF_TOKEN not found in .env file. Proceeding without login.")

    # Load Model & Tokenizer
    print(f"Loading base model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_len,
        dtype=None,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    print("Model and tokenizer loaded successfully.")

    # Load Local Dataset
    print("Loading local dataset files...")

    train_path = "data/medical_dialogue_train.csv"
    val_path = "data/medical_dialogue_validation.xlsx"
    test_path = "data/medical_dialogue_test.xlsx"

    train_df = pd.read_csv(train_path)
    val_df = pd.read_excel(val_path)
    test_df = pd.read_excel(test_path)

    # Ensure column names are consistent
    expected_cols = {"dialogue", "soap"}
    for df_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        if not expected_cols.issubset(df.columns):
            raise ValueError(f"{df_name} dataset must contain columns: {expected_cols}")

    # Convert pandas → HF Datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    # Format Dataset
    print("Applying chat formatting...")
    train_dataset = train_dataset.map(format_chat_template)
    val_dataset = val_dataset.map(format_chat_template)
    print("Dataset formatting complete.")

    # Configure LoRA
    print("Configuring LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    print("LoRA configuration complete.")

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
    )

    # Trainer
    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_len,
        packing=False,
        args=training_args,
    )

    # Train Model
    print("Starting fine-tuning...")
    trainer.train()
    print("Training completed successfully!")

    # Save Model
    print("Saving fine-tuned model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved locally at: {args.output_dir}")

    # Push to Hugging Face Hub
    if args.push_to_hub and hf_token:
        try:
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=args.push_to_hub, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=args.output_dir, repo_id=args.push_to_hub, repo_type="model")
            print(f"Successfully pushed to Hugging Face Hub: {args.push_to_hub}")
        except Exception as e:
            print(f"Error while pushing model: {e}")

# -----------------------------
# CLI Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a Gemma model for medical summarization using local dataset.")
    parser.add_argument("--model_name", type=str, default="unsloth/gemma-2b-it-bnb-4bit", help="Base model name.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing train/validation/test files.")
    parser.add_argument("--output_dir", type=str, default="./lora-gemma-medical-summarizer", help="Output directory for the fine-tuned model.")
    parser.add_argument("--push_to_hub", type=str, default=None, help="Optional: Push fine-tuned model to Hugging Face Hub.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device train batch size.")
    parser.add_argument("--grad_accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length.")
    args = parser.parse_args()

    main(args)
