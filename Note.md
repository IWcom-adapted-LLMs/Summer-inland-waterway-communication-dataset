<h1 align="center">🌊 LLIPSE</h1>
<p align="center">
  <i>Lightweight LLM‑backed Prediction Model for Inland Waterway Ship‑to‑Shore End‑to‑End Communications</i><br>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-MIT-green"></a>
  <a href="#"><img src="https://img.shields.io/badge/RTX4090-AutoDL-cyan"></a>
</p>

> **This repository contains the core code for the paper:**  
> *LLIPSE: Lightweight LLM‑backed Prediction Model for Inland Waterway Ship‑to‑Shore End‑to‑End Communications*  
## 📖 Overview

Accurate prediction of ship‑to‑shore communication is essential for intelligent autonomous navigation, but is severely hindered by limited real‑world data, stringent edge resource constraints, and the inability of existing methods to capture the physical mechanisms underlying channel dynamics.  

**LLIPSE** (Lightweight LLM‑backed Prediction Model for Inland Waterway Ship‑to‑Shore End‑to‑End Communications) addresses these challenges by leveraging a **large language model backbone** that is aggressively compressed via **ADMM 4‑bit quantization** and fine‑tuned with **EIB‑LoRA**. The model predicts **end‑to‑end latency** and **packet delivery ratio (PDR)** from real‑world data collected during summer inland waterway voyages.

> ✅ This repository contains the **official implementation** of the paper.  


---

## 🧠 Method Overview

| Stage | Technique | Purpose |
|-------|-----------|---------|
| **Quantization** | ADMM (Alternating Direction Method of Multipliers), layer‑wise, 4‑bit symmetric | Reduce model size while preserving accuracy. |
| **Fine‑tuning** | EIB‑LoRA (LoRA + learnable sparse gates) |  enables LLM to learn how environmental factors drive communication performanc.|
| **Evaluation** |Predict latency (ms) and PDR (%). |

All experiments run on a **single NVIDIA RTX 4090 (24 GB)**. The quantization pipeline processes each linear layer sequentially, avoiding out‑of‑memory issues even for 7B‑parameter models.


---

## 📁 Project Structure

| File | Role |
|------|------|
| `admm.py` | 🧊 Offline 4‑bit ADMM quantization → saves packed int8 weights |
| `train.py` | 🎓 Loads quantized backbone, adds EIB‑LoRA + regression head, trains with MSE + gate penalty |
| `test.py` | 🧪 Merges LoRA into quantized weights, evaluates MAE/RMSE/MAPE & inference time |
| `train_labeled.json` | 📊 Training data (conversation + delay) |
| `test_labeled.json` | 📊 Test data (same format) |

---

## 📊 Data Format

Each JSON file (`train_labeled.json`, `test_labeled.json`) 

## 🚀 Quick Start

### 1. Set paths in each script

| Script | Key variables (edit these) |
|--------|----------------------------|
| `admm.py` | `MODEL_PATH`, `QUANT_OUTPUT_DIR` |
| `train.py` | `ADMM_QUANTIZED_PATH`, `FULL_MODEL_PATH`, `TRAIN_DATA_PATH` (→ `train_labeled.json`), `TRAINING_OUTPUT_DIR` |
| `test.py` | `ADMM_QUANTIZED_PATH`, `FULL_MODEL_PATH`, `TRAINING_OUTPUT_DIR`, `TEST_DATA_PATH` (→ `test_labeled.json`) |

### 2. Run quantization

    python admm.py 
📦 Output – QUANT_OUTPUT_DIR/:

admm_quantized_weights.pt – packed int4 weights + scales

quantization_config.json – meta info

config.json, tokenizer files


### 3. Train EIB‑LoRA + regression head
bash
python train2.py
🎓 Output – TRAINING_OUTPUT_DIR/:

lora_and_reghead.pt – trained LoRA + reg head

target_norm.json – mean/std for denormalization

tokenizer files

### 4. Evaluate on test set
bash
python test.py
📈 Output – console metrics + test_predictions_final.jsonl (per‑sample predictions & latencies)

### 🧪 Example Metrics (after test.py)
text
MAE : ms
RMSE: ms
MAPE: 
Avg inference time: 
Compression ratio: ~5.6x
### ⚙️ Key Parameters (as used in the paper)
Parameter	Value	Description
ADMM_NBITS	4	Quantization bits
ADMM_ITER	50	ADMM iterations per layer
lora_r (rank)	16	LoRA rank
lora_alpha	32	LoRA scaling factor
lora_dropout	0.2	Dropout rate for LoRA
IB_LAMBDA	0.01	Sparsity penalty for EIB gates
num_train_epochs	6	Training epochs
learning_rate	5e-4	Learning rate
📦 Dependencies
   read requirements
### 💡 Important Notes
🧩 Layer‑wise quantization → each layer is sent to GPU, quantized, and immediately offloaded. No OOM even on 24GB GPU (tested on RTX 4090).

🚫 lm_head is not quantized (kept in fp16/bf16) to preserve generation quality.

🔄 Test script merges LoRA into quantized weights → single nn.Linear forward pass, no LoRA overhead.

🧮 EIB gates are absorbed into lora_A after training: lora_A = sigmoid(gate) * lora_A.

📄 License
This project is released under the MIT License.
See the LICENSE file for the full text.
Dependencies: PyTorch (BSD-style), Transformers (Apache 2.0), Datasets (Apache 2.0), NumPy (BSD). All are compatible with MIT.
