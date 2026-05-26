import os
import re
import json
import time
import datetime
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoConfig,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
)
from datasets import load_dataset

# ================= 计时辅助 =================
def format_duration(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

script_start_time = time.time()

# ================= 配置 =================
ADMM_QUANTIZED_PATH = "/root/autodl-tmp/admm-qwen"          # 量化权重目录
FULL_MODEL_PATH = "/root/autodl-tmp/Qwen3"                  # 原始全精度模型（提供 embedding/LayerNorm）
TRAIN_DATA_PATH = "/root/autodl-tmp/data/train.json"
TRAINING_OUTPUT_DIR = "/root/autodl-tmp/quanti-QwenMEAND"
DEVICE = "cuda:0"
IB_LAMBDA = 0.01

print(f"=== 配置 ===")
print(f"量化权重路径: {ADMM_QUANTIZED_PATH}")
print(f"原始模型路径: {FULL_MODEL_PATH}")
print(f"训练数据: {TRAIN_DATA_PATH}")
print(f"输出目录: {TRAINING_OUTPUT_DIR}")
print(f"IB lambda: {IB_LAMBDA}")
print("=" * 30)

# ================= 量化层定义（带 LoRA）=================
class QuantizedLinearWithLoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_r

        # 量化基座（冻结）
        self.register_buffer("weight_packed", torch.empty(0, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        # LoRA 参数（可训练）
        self.lora_A = nn.Parameter(torch.zeros(lora_r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, lora_r))
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_quantized_state(self, packed, scale, bias=None):
        self.weight_packed = packed.clone().detach().to(torch.int8)
        self.scale = scale.clone().detach().to(torch.float32)
        if bias is not None and self.bias is not None:
            self.bias.data = bias.clone().detach().to(torch.float32)

    def dequantize_weight(self) -> torch.Tensor:
        """解包：uint 0..15 -> int -8..7，再反量化"""
        packed = self.weight_packed
        low = (packed & 0x0F).to(torch.int8)
        high = ((packed >> 4) & 0x0F).to(torch.int8)
        # 正确映射：uint 值减 8 得到有符号值
        low = low - 8
        high = high - 8
        q_int = torch.zeros(packed.shape[0] * 2, dtype=torch.int8, device=packed.device)
        q_int[0::2] = low
        q_int[1::2] = high
        weight = q_int.to(torch.float32) * self.scale
        weight = weight.reshape(self.out_features, self.in_features)
        return weight.to(torch.bfloat16)

    def forward(self, x: torch.Tensor):
        # 反量化基座权重
        weight = self.dequantize_weight()
        base_out = F.linear(x.to(weight.dtype), weight, self.bias)

        # LoRA 路径
        lora_A = self.lora_A.to(x.dtype)
        lora_mid = self.lora_dropout(x) @ lora_A.T
        if hasattr(self, 'ib_gate') and self.ib_gate is not None:
            gate = torch.sigmoid(self.ib_gate).to(lora_mid.device)
            lora_mid = lora_mid * gate.to(x.dtype)
        lora_B = self.lora_B.to(x.dtype)
        lora_out = (lora_mid @ lora_B.T) * self.scaling
        return base_out + lora_out

# ================= 辅助函数 =================
def replace_linears_with_quantized_lora(model, lora_r=8, lora_alpha=16, lora_dropout=0.05):
    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            new_layer = QuantizedLinearWithLoRA(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
            )
            setattr(model, name, new_layer)
        else:
            replace_linears_with_quantized_lora(child, lora_r, lora_alpha, lora_dropout)

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

    for key, tensor in state_dict.items():
        if key.endswith("weight_packed"):
            layer_name = key.replace(".weight_packed", "")
            if layer_name in quant_layers:
                scale = state_dict.get(key.replace("weight_packed", "scale"))
                bias = state_dict.get(key.replace("weight_packed", "bias"))
                quant_layers[layer_name].set_quantized_state(tensor, scale, bias)
    print(f"已加载 {len(quant_layers)} 个量化层权重")

# ================= 加载模型 =================
print(">>> 加载 tokenizer 和 config...")
tokenizer = AutoTokenizer.from_pretrained(ADMM_QUANTIZED_PATH, trust_remote_code=True)
config = AutoConfig.from_pretrained(ADMM_QUANTIZED_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f">>> 从原始模型加载完整权重: {FULL_MODEL_PATH}")
base_model = AutoModelForCausalLM.from_pretrained(
    FULL_MODEL_PATH,
    config=config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cpu",
    trust_remote_code=True
)

print(">>> 替换 Linear 层为 QuantizedLinearWithLoRA...")
replace_linears_with_quantized_lora(base_model, lora_r=8, lora_alpha=16, lora_dropout=0.05)

print(">>> 加载 ADMM 量化权重...")
state_path = os.path.join(ADMM_QUANTIZED_PATH, "admm_quantized_weights.pt")
state_dict = torch.load(state_path, map_location="cpu")
load_quantized_state_into_model(base_model, state_dict)

# 冻结基座（所有参数），然后只放开 LoRA
for param in base_model.parameters():
    param.requires_grad = False
for name, param in base_model.named_parameters():
    if "lora_A" in name or "lora_B" in name:
        param.requires_grad = True
print(">>> 量化基座已冻结，LoRA 可训练。")
base_model.to(DEVICE)

# ================= 回归头 =================
class LLMWithRegressionHead(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base_model = base_model
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-9)
        logits = self.reg_head(pooled)
        loss = F.mse_loss(logits, labels) if labels is not None else None
        return RegressionOutput(logits, loss, outputs.hidden_states)

class RegressionOutput:
    def __init__(self, logits, loss, hidden_states):
        self.logits = logits
        self.loss = loss
        self.hidden_states = hidden_states

hidden_size = config.hidden_size
model_with_head = LLMWithRegressionHead(base_model, hidden_size)
model_with_head.to(DEVICE)

# ================= IB 门控注入 =================
print(">>> 注入 IB 门控...")
ib_gate_modules = []
for name, module in model_with_head.base_model.named_modules():
    if isinstance(module, QuantizedLinearWithLoRA):
        ib_gate = nn.Parameter(torch.zeros(module.lora_r))
        module.register_parameter('ib_gate', ib_gate)
        ib_gate_modules.append(module)
print(f"已注入 {len(ib_gate_modules)} 个门控")

# ================= 数据集处理 =================
print(f"加载数据: {TRAIN_DATA_PATH}")
dataset = load_dataset("json", data_files=TRAIN_DATA_PATH, split="train")

def parse_latency(text):
    m = re.search(r"Average delay is ([\d.]+)\s*ms", text)
    return float(m.group(1)) if m else float('nan')

def build_prompt(conversations):
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

targets_raw = []
for sample in dataset:
    convs = sample["conversations"]
    last = convs[-1] if convs else None
    delay = parse_latency(last["value"]) if (last and last["from"]=="gpt") else float('nan')
    targets_raw.append(delay)
dataset = dataset.add_column("target", targets_raw)
dataset = dataset.filter(lambda x: not torch.tensor(x["target"]).isnan())
clean_targets = [x["target"] for x in dataset]
target_mean, target_std = np.mean(clean_targets), np.std(clean_targets)
print(f"延迟统计: mean={target_mean:.2f}ms, std={target_std:.2f}ms")
if target_std < 1e-6:
    raise ValueError("标准差为0，无法归一化")
os.makedirs(TRAINING_OUTPUT_DIR, exist_ok=True)
with open(os.path.join(TRAINING_OUTPUT_DIR, "target_norm.json"), "w") as f:
    json.dump({"mean": target_mean, "std": target_std}, f)

def normalize(example):
    example["target_norm"] = (example["target"] - target_mean) / target_std
    return example
dataset = dataset.map(normalize)

def tokenize_fn(examples):
    texts, labels = [], []
    for conv_list, tnorm in zip(examples["conversations"], examples["target_norm"]):
        prompt = build_prompt(conv_list)
        if not prompt:
            continue
        texts.append(prompt)
        labels.append([tnorm])
    if not texts:
        return {"input_ids": [], "attention_mask": [], "labels": []}
    tok = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
    tok["labels"] = torch.tensor(labels, dtype=torch.float32)
    return tok

tokenized_datasets = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)
tokenized_datasets = tokenized_datasets.filter(lambda x: len(x["input_ids"]) > 0)
print(f"有效样本数: {len(tokenized_datasets)}")
if len(tokenized_datasets) == 0:
    raise ValueError("无有效样本")

# ================= 自定义 Trainer（修正接口） =================
class IBTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        task_loss = outputs.loss
        ib_penalty = 0.0
        for name, param in model.named_parameters():
            if 'ib_gate' in name:
                ib_penalty += torch.sigmoid(param).mean()
        total_loss = task_loss + IB_LAMBDA * ib_penalty
        return (total_loss, outputs) if return_outputs else total_loss

training_args = TrainingArguments(
    output_dir=TRAINING_OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    max_grad_norm=1.0,
    logging_steps=10,
    bf16=True,
    num_train_epochs=6,
    save_strategy="no",
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
)

trainer = IBTrainer(
    model=model_with_head,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# ================= 训练前诊断 =================
print(">>> 运行训练前诊断...")
test_loader = trainer.get_train_dataloader()
batch = next(iter(test_loader))
batch = {k: v.to(DEVICE) for k, v in batch.items()}
labels = batch.pop("labels")
model_with_head.train()
with torch.autograd.detect_anomaly():
    outputs = model_with_head(**batch, labels=labels)
    loss = outputs.loss
    print(f"诊断 loss = {loss.item():.6f}")
    if torch.isnan(loss):
        raise RuntimeError("Loss 为 NaN，检查模型输入或权重")
    loss.backward()
    grad_norm = 0.0
    for p in model_with_head.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm().item()**2
    grad_norm = math.sqrt(grad_norm)
    print(f"梯度范数 = {grad_norm:.6f}")
    model_with_head.zero_grad()
print("诊断通过，开始正式训练...")

# ================= 训练 =================
train_start = time.time()
trainer.train()
train_duration = time.time() - train_start
print(f"训练耗时: {format_duration(train_duration)}")

# ================= 吸收门控并保存 =================
print(">>> 吸收门控到 LoRA 权重...")
with torch.no_grad():
    for module in ib_gate_modules:
        if hasattr(module, 'ib_gate') and module.ib_gate is not None:
            g = torch.sigmoid(module.ib_gate).to(module.lora_A.device)
            module.lora_A.data = module.lora_A.data * g.unsqueeze(1)
            del module.ib_gate
            module.ib_gate = None

print(f">>> 保存模型到 {TRAINING_OUTPUT_DIR}")
lora_state = {n: p.detach().cpu() for n, p in model_with_head.named_parameters() if p.requires_grad}
torch.save(lora_state, os.path.join(TRAINING_OUTPUT_DIR, "lora_and_reghead.pt"))
torch.save(model_with_head.reg_head.state_dict(), os.path.join(TRAINING_OUTPUT_DIR, "reg_head.pt"))
tokenizer.save_pretrained(TRAINING_OUTPUT_DIR)
print("训练完成。")

print(f"总耗时: {format_duration(time.time() - script_start_time)}")