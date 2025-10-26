# Fine-Tuning LLM for Medical Chat Summarization

### Project Overview
This project focuses on fine-tuning a Gemma 2B large language model to accurately summarize medical dialogues into structured SOAP notes (Subjective, Objective, Assessment, Plan). 🩺 The core objective is to create a computationally efficient yet powerful model that can be deployed as a reliable API service.

To achieve this, the project leverages state-of-the-art techniques, including 4-bit quantization and LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning (PEFT), all accelerated by the Unsloth library. The final, optimized model is served via a Flask API, containerized with Docker, and designed for easy deployment on cloud platforms like Hugging Face Spaces.

## Model & Fine-Tuning

* **Base Model:** `unsloth/gemma-2b-it-bnb-4bit`
* **Technique:** 4-bit Quantized LoRA (PEFT)
* **Dataset:** `omi-health/medical-dialogue-to-soap-summary`

---

### Features
- Accurate SOAP Summarization: Converts unstructured doctor-patient conversations into organized SOAP notes.

- Efficient Performance: Utilizes 4-bit quantization and Unsloth to ensure fast inference and a low memory footprint, making it suitable for deployment on cost-effective hardware.

- State-of-the-Art Fine-Tuning: Employs LoRA to efficiently adapt a powerful base model without retraining all its parameters, saving significant time and resources.

- REST API for Integration: Provides a simple POST endpoint (/summarize) that accepts a dialogue and returns a JSON summary, ready for integration into other applications like EHR systems or telehealth platforms.

- Scalable Deployment: Packaged with a Dockerfile and Gunicorn for robust, production-ready deployment.

---

## ⚙️ How to Run

### 1. Create Environment
```bash
conda create -n FineTuneLLM python=3.11 -y
conda activate FineTuneLLM # To activate
conda deactivate # To deactivate
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```
### 3. Set API Key (Keep your all API in .env file) - best approach
Create a `.env` file in the project root:
```
HF_TOKEN = "your_HF_TOKEN_here"
```

---

## Project Structure
The project is organized into a modular structure to separate concerns, making it clean, maintainable, and easy to navigate.
```
Fine-Tuning LLM for Medical Chat Summarization/
├── .env                                # All API Key is stored here
├── .gitignore                          # Specifies files for Git to ignore
├── research/
│   └── fine-tuning-experiment.ipynb    # First I explored the full project here before writing moduler structure code
├── src/                                # Start Modular Coding
│   ├─ __init__.pt.py
│   ├─ app.py                           # Flask API for model serving and inference
│   ├─ evaluate.py                      # Script for quantitative (ROUGE, BERTScore) and qualitative evaluation
│   └─ train.py                         # Script for the end-to-end model fine-tuning process       
└── LICENSE
└── README.md                           # Comprehensive project documentation
└── requirements.txt                    # Lists all Python dependencies
└── setup.py
```

---

## Tech Stack
The project is built on a modern MLOps stack designed for efficiency and performance.

- Modeling & Training: PyTorch, Unsloth, Hugging Face (Transformers, PEFT, Datasets, TRL), bitsandbytes

- API & Serving: Flask, Gunicorn

- Deployment: Docker, Hugging Face Spaces

- Evaluation: evaluate, bert_score

---

## Example Output
The API takes a raw medical dialogue and transforms it into a structured SOAP note.

Input (POST /summarize)
```json
{
  "dialogue": "Doctor: Hello, what brings you in today? Patient: I've had a really bad headache behind my eyes for the last two days, and I feel nauseous. I took some Tylenol, but it didn't help much. I haven't had a fever. Doctor: Okay, I see. Let's check your blood pressure; it's 130/85. Your pupils are equal and reactive to light. Any history of migraines? Patient: No, never."
}
```
Output (Generated Summary)
```json
{
  "summary": "S: Patient is a female complaining of a headache behind the eyes for the last 2 days, accompanied by nausea. Reports taking Tylenol with minimal relief. Denies fever or a history of migraines.\nO: Blood pressure is 130/85. Pupils are equal, round, and reactive to light.\nA: Headache, likely tension-related or early-stage migraine.\nP: Recommend MRI of the brain to rule out other causes. Prescribe Sumatriptan for symptomatic relief. Advise patient to keep a headache diary and follow up in 1 week."
}
```
---

## Evaluation Results

| Metric | Score |
| :--- | :--- |
| ROUGE-1 | 39.07 |
| ROUGE-2 | 23.26 |
| ROUGE-L | 25.81 |
| rougeLsum | 35.72 |
| BERTScore F1 | 86.71 |

---

## ✍️ Author
Developed by **Aslam Sikder**, October 2025  
Email: [aslamsikder.edu@gmail.com](mailto:aslamsikder.edu@gmail.com)  
LinkedIn: [Aslam Sikder - Linkedin](https://www.linkedin.com/in/aslamsikder)  
Google Scholar: [Aslam Sikder - Google Scholar](https://scholar.google.com/citations?hl=en&user=Ip1qQi8AAAAJ)
