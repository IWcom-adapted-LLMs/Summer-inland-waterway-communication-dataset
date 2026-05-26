<h1 align="center">🌊 LLIPSE</h1>
<p align="center">
  <i>Lightweight LLM‑backed Prediction Model for Inland Waterway Ship‑to‑Shore End‑to‑End Communications</i><br>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="#"><img src="https://img.shields.io/badge/RTX4090-AutoDL-cyan"></a>
</p>

> **This repository contains the core code for the paper:**  
> *LLIPSE: Lightweight LLM‑backed Prediction Model for Inland Waterway Ship‑to‑Shore End‑to‑End Communications*  
> Implementation of ADMM 4‑bit quantization + EIB‑LoRA fine‑tuning for latency prediction.

---

## 🚀 What is this?

This repo provides a **lightweight yet powerful** recipe to:
- 📦 **Compress LLMs** (like Qwen‑7B) to **4‑bit** using ADMM (layer‑wise, OOM‑safe).
- 🎯 **Fine‑tune** with **EIB‑LoRA** (sparse gates + LoRA) for regression tasks.
- ⏱️ **Predict conversation delay** (ms) with high accuracy and low inference overhead.

> ✅ Tested on **AutoDL RTX 4090 (24GB)** – quantizes 7B models without OOM.

---

## 📁 Project Structure

| File | Role |
|------|------|
| `admm.py` | 🧊 Offline 4‑bit ADMM quantization → saves packed int8 weights |
| `train2.py` | 🎓 Loads quantized backbone, adds EIB‑LoRA + regression head, trains with MSE + gate penalty |
| `test.py` | 🧪 Merges LoRA into quantized weights, evaluates MAE/RMSE/MAPE & inference time |
| `train_labeled.json` | 📊 Training data (conversation + delay) |
| `test_labeled.json` | 📊 Test data (same format) |

---

## 🧾 Data Format

Your JSON file should look like this (one object per sample):

```json
{
  "conversations": [
    {"from": "human", "value": "What's the average delay?"},
    {"from": "gpt", "value": "The average delay is 234.56 ms"}
  ]
}
The script extracts the number using regex: Average delay is ([\d.]+)\s*ms
Samples without a match are automatically skipped.

⚡ Quick Start
1️⃣ Set paths in each script
Script	Key variables (edit these)
admm.py	MODEL_PATH, QUANT_OUTPUT_DIR
train2.py	ADMM_QUANTIZED_PATH, FULL_MODEL_PATH, TRAIN_DATA_PATH (→ train_labeled.json), TRAINING_OUTPUT_DIR
test.py	ADMM_QUANTIZED_PATH, FULL_MODEL_PATH, TRAINING_OUTPUT_DIR, TEST_DATA_PATH (→ test_labeled.json)
2️⃣ Run quantization
bash
python admm.py
📦 Output – QUANT_OUTPUT_DIR/:

admm_quantized_weights.pt – packed int4 weights + scales

quantization_config.json – meta info

config.json, tokenizer files

3️⃣ Train EIB‑LoRA + regression head
bash
python train2.py
🎓 Output – TRAINING_OUTPUT_DIR/:

lora_and_reghead.pt – trained LoRA + reg head

target_norm.json – mean/std for denormalization

tokenizer files

4️⃣ Evaluate on test set
bash
python test.py
📈 Output – console metrics + test_predictions_final.jsonl (per‑sample predictions & latencies)

🧪 Example Metrics (after test.py)
text
MAE : 2.35 ms
RMSE: 3.51 ms
MAPE: 4.8%
R²  : 0.89
Avg inference time: 0.045 s/sample
Compression ratio: ~5.6x
🛠️ Key Parameters
Parameter	Default	What it does
ADMM_NBITS	4	Quantization bits
ADMM_ITER	50	ADMM iterations per layer
lora_r	8	LoRA rank
IB_LAMBDA	0.01	Sparsity penalty for EIB gates
num_train_epochs	6	Training epochs
learning_rate	5e-4	Learning rate
📦 Dependencies
bash
pip install torch transformers datasets numpy
💡 Important Notes
🧩 Layer‑wise quantization → each layer is sent to GPU, quantized, and immediately offloaded. No OOM even on 24GB GPU (tested on RTX 4090).

🚫 lm_head is not quantized (kept in fp16/bf16) to preserve generation quality.

🔄 Test script merges LoRA into quantized weights → single nn.Linear forward pass, no LoRA overhead.

🧮 EIB gates are absorbed into lora_A after training: lora_A = sigmoid(gate) * lora_A.

📄 License
This project is released under the MIT License.
See the LICENSE file for the full text.
Dependencies: PyTorch (BSD-style), Transformers (Apache 2.0), Datasets (Apache 2.0), NumPy (BSD). All are compatible with MIT.
