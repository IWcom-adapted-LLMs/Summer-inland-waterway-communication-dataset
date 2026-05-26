import os
import re
import json
import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from datasets import load_dataset

warnings.filterwarnings("ignore")

# ================= Paths (EDIT THESE) =================
ADMM_QUANTIZED_PATH = "/path/to/quantized/model"
FULL_MODEL_PATH = "/path/to/original/foundation/model"
TRAINING_OUTPUT_DIR = "/path/to/training/output"
TEST_DATA_PATH = "/path/to/test_labeled.json"
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

# Paper hyperparameters (must match training)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.2

class QuantizedLinearWithLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 lora_r: int = LORA_R, lora_alpha: int = LORA_ALPHA, lora_dropout: float = LORA_DROPOUT):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.scaling = lora_alpha / lora_r

        self.register_buffer("weight_packed", torch.empty(0, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.zeros(lora_r, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r, dtype=torch.float32))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def set_quantized_state(self, packed, scale, bias=None):
        self.weight_packed = packed.clone().detach().to(torch.int8)
        self.scale = scale.clone().detach().to(torch.float32)
        if bias is not None and self.bias is not None:
            self.bias.data = bias.clone().detach().to(torch.float32)

    def dequantize_weight(self) -> torch.Tensor:
        if self.weight_packed.numel() == 0:
            return torch.zeros(self.out_features, self.in_features, dtype=DTYPE, device=self.weight_packed.device)
        packed = self.weight_packed
        low = (packed & 0x0F).to(torch.int8) - 8
        high = ((packed >> 4) & 0x0F).to(torch.int8) - 8
        q_int = torch.empty(packed.shape[0] * 2, dtype=torch.int8, device=packed.device)
        q_int[0::2] = low
        q_int[1::2] = high
        scale = self.scale.to(device=packed.device)
        if scale.ndim == 1 and scale.shape[0] == self.out_features:
            weight = q_int.reshape(self.out_features, self.in_features).to(DTYPE)
            weight = weight * scale.unsqueeze(1)
        elif scale.ndim == 2 and scale.shape == (self.out_features, 1):
            weight = q_int.reshape(self.out_features, self.in_features).to(DTYPE)
            weight = weight * scale
        else:
            weight = q_int.to(DTYPE) * scale
            expected = self.out_features * self.in_features
            if weight.numel() == expected:
                weight = weight.reshape(self.out_features, self.in_features)
            elif weight.numel() == self.in_features * self.out_features:
                weight = weight.reshape(self.in_features, self.out_features).T
            else:
                raise RuntimeError("dequantize shape mismatch")
        return weight

    def forward(self, x: torch.Tensor):
        weight = self.dequantize_weight()
        target_dtype = x.dtype
        weight = weight.to(target_dtype)
        bias = self.bias.to(target_dtype) if self.bias is not None else None
        base_out = F.linear(x, weight, bias)
        lora_A = self.lora_A.to(target_dtype)
        lora_B = self.lora_B.to(target_dtype)
        lora_mid = self.lora_dropout(x) @ lora_A.T
        lora_out = (lora_mid @ lora_B.T) * self.scaling
        return base_out + lora_out

def replace_linears_with_quantized_lora(model, skip_names=None):
    if skip_names is None:
        skip_names = ['lm_head']
    for name, child in model.named_children():
        if any(skip in name for skip in skip_names):
            continue
        if isinstance(child, nn.Linear):
            new_layer = QuantizedLinearWithLoRA(
                child.in_features, child.out_features,
                bias=child.bias is not None
            )
            setattr(model, name, new_layer)
        else:
            replace_linears_with_quantized_lora(child, skip_names)

def load_quantized_state_into_model(model, state_dict):
    quant_layers = {}
    def collect(module, prefix=""):
        for name, child in module.named_children():
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, QuantizedLinearWithLoRA):
                quant_layers[full] = child
            else:
                collect(child, full)
    collect(model)
    loaded = 0
    for layer_name, layer in quant_layers.items():
        packed_key = f"{layer_name}.weight_packed"
        scale_key = f"{layer_name}.scale"
        bias_key = f"{layer_name}.bias"
        if packed_key in state_dict and scale_key in state_dict:
            packed = state_dict[packed_key]
            scale = state_dict[scale_key]
            bias = state_dict.get(bias_key, None)
            layer.set_quantized_state(packed, scale, bias)
            loaded += 1
    print(f"Loaded {loaded}/{len(quant_layers)} quantized layers")

class RegressionOutput:
    def __init__(self, logits, loss, hidden_states):
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states

class LLMWithRegressionHead(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        logits = self.reg_head(pooled)
        loss = None
        if labels is not None:
            loss = F.mse_loss(logits, labels.view(-1, 1))
        return RegressionOutput(logits, loss, outputs.hidden_states)


norm_file = os.path.join(TRAINING_OUTPUT_DIR, "target_norm.json")
with open(norm_file, "r") as f:
    norm_params = json.load(f)
target_mean = norm_params["mean"]
target_std = norm_params["std"]
print(f"mean={target_mean:.2f} ms, std={target_std:.2f} ms")


tokenizer = AutoTokenizer.from_pretrained(TRAINING_OUTPUT_DIR, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


config = AutoConfig.from_pretrained(ADMM_QUANTIZED_PATH, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    FULL_MODEL_PATH,
    config=config,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=True
)

replace_linears_with_quantized_lora(base_model, skip_names=['lm_head'])

state_dict = torch.load(os.path.join(ADMM_QUANTIZED_PATH, "admm_quantized_weights.pt"), map_location="cpu")
load_quantized_state_into_model(base_model, state_dict)

hidden_size = config.hidden_size
model_with_head = LLMWithRegressionHead(base_model, hidden_size)

lora_state = torch.load(os.path.join(TRAINING_OUTPUT_DIR, "lora_and_reghead.pt"), map_location="cpu")
model_with_head.load_state_dict(lora_state, strict=False)
model_with_head.to(DEVICE)
model_with_head.eval()


def parse_latency(text):
    m = re.search(r"Average delay is ([\d.]+)\s*ms", text)
    return float(m.group(1)) if m else None

def build_prompt_only_human(conversations):
    messages = [{"role": "user", "content": turn["value"]} for turn in conversations if turn["from"] == "human"]
    if not messages:
        return ""
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        prompt = ""
        for msg in messages:
            prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

@torch.no_grad()
def predict_delay_norm(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(DEVICE)
    outputs = model_with_head(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
    return outputs.logits.item()

def predict_delay(text):
    return predict_delay_norm(text) * target_std + target_mean


test_dataset = load_dataset("json", data_files=TEST_DATA_PATH, split="train")
true_delays, prompts = [], []
for sample in test_dataset:
    convs = sample["conversations"]
    last = convs[-1]
    if last["from"] != "gpt":
        continue
    delay = parse_latency(last["value"])
    if delay is None:
        continue
    prompt = build_prompt_only_human(convs)
    if not prompt:
        continue
    true_delays.append(delay)
    prompts.append(prompt)

print(f"Test samples: {len(prompts)}")
if len(prompts) == 0:
    raise ValueError("No valid test samples")


import time
pred_delays, inference_times = [], []
for i, prompt in enumerate(prompts):
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()
    pred = predict_delay(prompt)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    inference_times.append(elapsed)
    pred_delays.append(pred)
    if i < 10:
        print(f"Sample {i+1}: true={true_delays[i]:.2f} ms, pred={pred:.2f} ms, time={elapsed*1000:.2f} ms")

avg_time = np.mean(inference_times) * 1000
std_time = np.std(inference_times) * 1000
print(f"Inference time: mean={avg_time:.2f} ms, std={std_time:.2f} ms")


true_arr = np.array(true_delays)
pred_arr = np.array(pred_delays)
ae = np.abs(true_arr - pred_arr)
mse = np.mean(ae**2)
mae = np.mean(ae)
rmse = np.sqrt(mse)
nonzero = true_arr != 0
mape = (100.0 * np.mean(ae[nonzero] / np.abs(true_arr[nonzero]))) if np.any(nonzero) else 0.0
denom = np.abs(true_arr) + np.abs(pred_arr)
valid = denom > 0
smape = (100.0 * np.mean(2 * ae[valid] / denom[valid])) if np.any(valid) else 0.0
ss_res = np.sum(ae**2)
ss_tot = np.sum((true_arr - np.mean(true_arr))**2)
r2 = 1 - ss_res/ss_tot if ss_tot != 0 else 0.0

print("\n" + "="*40)
print(f"MSE: {mse:.4f} ms²")
print(f"MAE: {mae:.4f} ms")
print(f"RMSE: {rmse:.4f} ms")
print(f"MAPE: {mape:.2f}%")
print(f"SMAPE: {smape:.2f}%")
print(f"R²: {r2:.4f}")
print("="*40)


out_file = os.path.join(TRAINING_OUTPUT_DIR, "test_predictions_quantized_lora.txt")
with open(out_file, "w") as f:
    f.write(f"Avg inference time: {avg_time:.2f} ms (std {std_time:.2f} ms)\n")
    f.write(f"MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2f}%, SMAPE={smape:.2f}%, R²={r2:.4f}\n\n")
    for i, (t, p, et) in enumerate(zip(true_delays, pred_delays, inference_times)):
        f.write(f"Sample {i+1}: true={t:.2f} ms, pred={p:.2f} ms, time={et*1000:.2f} ms\n")
print(f"Results saved to {out_file}")
