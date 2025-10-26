import os
import torch
import argparse
import pandas as pd
from tqdm import tqdm
import evaluate
from unsloth import FastLanguageModel

# -----------------------------
# Global Model Configuration
# -----------------------------
MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True


# -----------------------------
# Load Model Function
# -----------------------------
def load_model_for_eval(model_name):
    """Loads a model (base or fine-tuned) for evaluation."""
    print(f"\nLoading model: {model_name} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
        trust_remote_code=True,
        cache_dir="./models"
    )
    model.to("cuda")
    model.eval()
    print("Model loaded and set to eval mode.")
    return model, tokenizer


# -----------------------------
# Format Prompt Function
# -----------------------------
def format_inference_prompt(dialogue, tokenizer):
    """Formats the input dialogue with the same chat template used in training."""
    messages = [
        {
            "role": "user",
            "content": f"Summarize the following medical dialogue into a SOAP note:\n\n{dialogue}"
        }
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return prompt


# -----------------------------
# Text Generation Function
# -----------------------------
def generate_summary(model, tokenizer, prompt):
    """Generates a summary using the model."""
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).to(torch.long)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=attention_mask,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    # Extract model response
    parts = generated_text.split("<start_of_turn>model\n")
    if len(parts) > 1:
        return parts[-1].strip()
    else:
        return generated_text.strip()


# -----------------------------
# Main Evaluation Function
# -----------------------------
def main(args):
    # --- Load Local Test Data ---
    test_path = os.path.join("data", "medical_dialogue_test.xlsx")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Could not find test file at: {test_path}")

    print(f"\nLoading local dataset: {test_path}")
    test_df = pd.read_excel(test_path) if test_path.endswith(".xlsx") else pd.read_csv(test_path)

    required_cols = {"dialogue", "soap"}
    if not required_cols.issubset(set(test_df.columns)):
        raise ValueError(f"Dataset must contain columns: {required_cols}")

    # --- Load Base and Fine-Tuned Models ---
    base_model, base_tokenizer = load_model_for_eval(args.base_model_name)
    ft_model, ft_tokenizer = load_model_for_eval(args.finetuned_model_name)

    # --- Qualitative Comparison ---
    print("\n--- QUALITATIVE COMPARISON ---")
    results = []
    samples = test_df.sample(n=min(args.num_qualitative_samples, len(test_df)), random_state=42)

    for idx, row in samples.iterrows():
        dialogue = row["dialogue"]
        reference = row["soap"]

        base_prompt = format_inference_prompt(dialogue, base_tokenizer)
        ft_prompt = format_inference_prompt(dialogue, ft_tokenizer)

        base_summary = generate_summary(base_model, base_tokenizer, base_prompt)
        ft_summary = generate_summary(ft_model, ft_tokenizer, ft_prompt)

        print(f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"SAMPLE {idx+1}")
        print(f"\nDIALOGUE:\n{dialogue}\n")
        print(f"REFERENCE SOAP:\n{reference}\n")
        print(f"BASELINE MODEL SUMMARY:\n{base_summary}\n")
        print(f"FINE-TUNED MODEL SUMMARY:\n{ft_summary}\n")

        results.append({
            "dialogue": dialogue,
            "reference": reference,
            "baseline_summary": base_summary,
            "finetuned_summary": ft_summary
        })

    pd.DataFrame(results).to_csv("qualitative_evaluation_results.csv", index=False)
    print("\nSaved qualitative results → qualitative_evaluation_results.csv")

    # --- Quantitative Evaluation ---
    print("\n--- QUANTITATIVE EVALUATION ---")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    predictions, references = [], []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        dialogue = row["dialogue"]
        reference = row["soap"]
        prompt = format_inference_prompt(dialogue, ft_tokenizer)
        prediction = generate_summary(ft_model, ft_tokenizer, prompt)

        predictions.append(prediction)
        references.append(reference)

    # Compute Metrics
    rouge_results = rouge.compute(predictions=predictions, references=references)
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")

    print("\n--- QUANTITATIVE EVALUATION RESULTS ---")
    print("\nROUGE Scores:")
    for key, value in rouge_results.items():
        print(f"{key}: {value * 100:.2f}")

    avg_bert_f1 = sum(bert_results["f1"]) / len(bert_results["f1"])
    print(f"\nBERTScore:\nAverage F1 Score: {avg_bert_f1 * 100:.2f}")

    print("\nEvaluation complete!")

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate base and fine-tuned Gemma models on local dataset.")
    parser.add_argument("--base_model_name", type=str, default="unsloth/gemma-2b-it-bnb-4bit",
                        help="Base model name from Hugging Face Hub.")
    parser.add_argument("--finetuned_model_name", type=str, required=True,
                        help="Path or Hub ID of fine-tuned LoRA model (e.g., './lora-gemma-medical-summarizer').")
    parser.add_argument("--num_qualitative_samples", type=int, default=3,
                        help="Number of qualitative samples to display.")

    args = parser.parse_args()
    main(args)
