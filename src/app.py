from flask import Flask, request, jsonify
import torch
import os
from unsloth import FastLanguageModel

# -----------------------------
# Flask App Initialization
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Hugging Face Model Config
# -----------------------------
# 👇 Change this to your own model repo on Hugging Face
HF_MODEL_ID = "aslamsikder/lora-fine-tuned-gemma-2b-medical-dialogue-to-soap-summary"
MODEL_ID = os.environ.get("MODEL_ID", HF_MODEL_ID)

# Optional: If your model is private, set your Hugging Face token in environment
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Global model and tokenizer
model, tokenizer = None, None


# -----------------------------
# Model Loading
# -----------------------------
def load_model():
    """Loads the fine-tuned model from Hugging Face."""
    global model, tokenizer
    try:
        print(f"Loading fine-tuned model from Hugging Face: {MODEL_ID} ...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
            token=HF_TOKEN,   # Optional for private models
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        FastLanguageModel.for_inference(model)

        print(f"Model loaded successfully from Hugging Face ({device.upper()})")
    except Exception as e:
        print(f"Error loading model: {e}")
        model, tokenizer = None, None


# -----------------------------
# Summarization Endpoint
# -----------------------------
@app.route("/summarize", methods=["POST"])
def summarize():
    """Summarize medical dialogues into SOAP notes."""
    global model, tokenizer
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded. Try again later."}), 500

    # Parse input
    data = request.get_json(silent=True)
    if not data or "dialogue" not in data:
        return jsonify({"error": "Missing key: 'dialogue'"}), 400

    dialogue = data["dialogue"].strip()
    if not dialogue:
        return jsonify({"error": "Empty dialogue text."}), 400

    # Format as chat prompt
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

    # Prepare model input
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    attention_mask = (inputs["input_ids"] != tokenizer.pad_token_id).long()

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # Clean and extract
        if "<start_of_turn>model" in generated_text:
            summary = generated_text.split("<start_of_turn>model")[-1].strip()
        else:
            summary = generated_text.strip()

        summary = summary.replace("### Response:", "").strip()

        return jsonify({
            "summary": summary,
            "model_id": MODEL_ID
        })

    except Exception as e:
        return jsonify({"error": f"Generation failed: {str(e)}"}), 500


# -----------------------------
# Health Check
# -----------------------------
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({
        "status": "API running",
        "model_loaded": model is not None,
        "model_id": MODEL_ID
    })


# -----------------------------
# Run the App
# -----------------------------
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
