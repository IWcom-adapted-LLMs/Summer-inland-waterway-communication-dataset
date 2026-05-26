#!/usr/bin/env python3
"""
ADMM 4-bit Quantization Script – 离线逐层量化版本（防 OOM）
- 原始模型全程驻留在 CPU，每层权重单独送 GPU 量化后立即移出
- 跳过 lm_head 层（大词汇表投影，量化得不偿失）
- 输出磁盘压缩比与量化统计
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json, math, glob, torch, torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple

# ========== 配置 ==========
MODEL_PATH = "/root/autodl-tmp/Qwen2-7b"
QUANT_OUTPUT_DIR = "/root/autodl-tmp/admm-qwen2-7b"
ADMM_NBITS = 4
ADMM_RHO = 0.01
ADMM_ITER = 50
DTYPE = torch.bfloat16
os.makedirs(QUANT_OUTPUT_DIR, exist_ok=True)

# ========== 工具函数 ==========
def format_bytes(b):
    for unit in ['B','KB','MB','GB','TB']:
        if b < 1024.0: return f"{b:.2f} {unit}"
        b /= 1024.0
    return f"{b:.2f} PB"

def dir_size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files: total += os.path.getsize(os.path.join(root, f))
    return total

def model_file_size(model_path):
    files = glob.glob(os.path.join(model_path, "*.safetensors")) + \
            glob.glob(os.path.join(model_path, "*.bin"))
    return sum(os.path.getsize(f) for f in files) if files else None

# ========== ADMM 量化核心（临时 GPU 对象）==========
class ADMMQuantizer:
    """单层权重临时量化器，所有数据在 GPU 上完成 ADMM 迭代"""
    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor], nbits=4, rho=0.01):
        self.in_features = weight.shape[1]
        self.out_features = weight.shape[0]
        self.nbits = nbits
        self.rho = rho
        self.device = weight.device

        # 初始化 ADMM 变量（全部 float32，更精确）
        self.W_fp = weight.to(torch.float32).clone()
        self.W_q = self.W_fp.clone()
        self.Z = self.W_fp.clone()
        self.Lambda = torch.zeros_like(self.W_fp)
        self.bias = bias.to(torch.float32).clone() if bias is not None else None

        # 确定缩放因子
        max_abs = torch.max(torch.abs(self.W_fp))
        self.scale = max_abs / (2**(nbits-1) - 1)
        if self.scale.item() == 0:
            self.scale = torch.tensor(1.0, device=self.device)

    def project_to_discrete(self, X):
        s = self.scale
        x = X / s
        q_min = -(2**(self.nbits-1))
        q_max = (2**(self.nbits-1)) - 1
        q = torch.clamp(torch.round(x), q_min, q_max)
        return q * s

    def run_admm(self, iterations=50):
        for it in range(iterations):
            # 更新 W_q
            new_W_q = (self.W_fp + self.rho * self.Z - self.Lambda) / (1 + self.rho)
            self.W_q.copy_(new_W_q)
            # 投影到离散集合
            X = self.W_q + self.Lambda / self.rho
            Z_new = self.project_to_discrete(X)
            self.Z.copy_(Z_new)
            # 更新对偶变量
            self.Lambda.add_(self.rho * (self.W_q - self.Z))
        # 返回量化后的 Z 和 scale（均在 GPU 上）
        return self.Z, self.scale, self.bias

# ========== 离线逐层量化 ==========
def quantize_offline(model, num_iter=50, rho=0.01, verbose=True):
    """
    遍历模型中所有 nn.Linear（跳过 lm_head），
    将单层权重暂时送入 GPU 量化，结果存入 CPU 列表。
    """
    quantized_data = {}
    shape_info = {}

    for name, module in model.named_modules():
        if name == "lm_head":
            if verbose: print(f"Skipping layer: {name} (large vocab, kept fp16)")
            continue
        if isinstance(module, nn.Linear):
            if verbose:
                print(f"Quantizing: {name}  [{module.in_features}, {module.out_features}]")

            # 1. 取出权重和偏置，立即移到 GPU
            w = module.weight.data.detach()
            b = module.bias.data.detach() if module.bias is not None else None
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            w_gpu = w.to(device)
            b_gpu = b.to(device) if b is not None else None

            # 2. 创建临时量化器并运行 ADMM
            quantizer = ADMMQuantizer(w_gpu, b_gpu, nbits=ADMM_NBITS, rho=rho)
            Z, scale, bias_quant = quantizer.run_admm(iterations=num_iter)

            # 3. 将结果移至 CPU 并保存
            quantized_data[name] = {
                "Z": Z.cpu(),
                "scale": scale.cpu(),
                "bias": bias_quant.cpu() if bias_quant is not None else None
            }
            shape_info[name] = {
                "weight_shape": list(Z.shape),
                "bias_shape": list(bias_quant.shape) if bias_quant is not None else None
            }

            # 4. 立即清理 GPU 显存
            del w_gpu, b_gpu, quantizer, Z, scale, bias_quant
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if verbose:
                print(f"  -> Done.")

    if verbose:
        print(f"Total quantized layers: {len(quantized_data)}")
    return quantized_data, shape_info

# ========== 打包 & 保存 ==========
def pack_int4_weights(z: torch.Tensor, scale: torch.Tensor):
    q_int = torch.round(z / scale).clamp(-8, 7).to(torch.int8)
    original_numel = q_int.numel()
    q_uint = (q_int + 8).to(torch.uint8)
    q_flat = q_uint.reshape(-1)
    if q_flat.shape[0] % 2 == 1:
        q_flat = torch.cat([q_flat, torch.tensor([0], dtype=torch.uint8)])
    packed = (q_flat[0::2]) | (q_flat[1::2] << 4)
    return packed.to(torch.int8), original_numel

def save_quantized_model(quantized_data, shape_info, output_dir, config, tokenizer):
    os.makedirs(output_dir, exist_ok=True)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    quant_state = {}
    final_shape_info = {}
    for layer_name, data in quantized_data.items():
        packed, orig_numel = pack_int4_weights(data["Z"], data["scale"])
        quant_state[f"{layer_name}.weight_packed"] = packed
        quant_state[f"{layer_name}.scale"] = data["scale"]
        final_shape_info[layer_name] = {
            "original_weight_shape": shape_info[layer_name]["weight_shape"],
            "original_numel": orig_numel,
            "packed_shape": list(packed.shape),
            "bias_shape": shape_info[layer_name]["bias_shape"]
        }
        if data["bias"] is not None:
            quant_state[f"{layer_name}.bias"] = data["bias"].to(torch.float16)

    with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
        json.dump({
            "nbits": ADMM_NBITS,
            "rho": ADMM_RHO,
            "iterations": ADMM_ITER,
            "format": "int4_packed_per_tensor_symmetric",
            "packing_method": "uint4_pair_little_endian",
            "shape_info": final_shape_info
        }, f, indent=2)
    torch.save(quant_state, os.path.join(output_dir, "admm_quantized_weights.pt"))
    print(f"Quantized weights saved to {output_dir}")

# ========== 主程序 ==========
if __name__ == "__main__":
    # 原始模型文件大小（用于计算压缩比）
    orig_size = model_file_size(MODEL_PATH)
    if orig_size is None:
        print("[Warn] Original model files not found, skipping disk compression ratio.")
    else:
        print(f"[Orig]  Original model file size: {format_bytes(orig_size)}")

    # 加载模型到 CPU（完全不占用 GPU）
    print(f"Loading full-precision model from {MODEL_PATH} (CPU only)...")
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        config=config,
        torch_dtype=DTYPE,
        device_map="cpu",          # 强制 CPU
        trust_remote_code=True
    )
    model.eval()
    print("Model loaded on CPU.")

    # 离线逐层量化
    print("Starting offline layerwise ADMM quantization...")
    quantized_data, shape_info = quantize_offline(model, num_iter=ADMM_ITER, rho=ADMM_RHO, verbose=True)

    # 保存量化结果
    save_quantized_model(quantized_data, shape_info, QUANT_OUTPUT_DIR, config, tokenizer)

    # 磁盘压缩比统计
    quant_size = dir_size(QUANT_OUTPUT_DIR)
    print("\n" + "="*50)
    print("✅ ADMM Quantization Statistics")
    print("="*50)
    if orig_size is not None:
        ratio = orig_size / quant_size
        print(f"Original model size (disk):   {format_bytes(orig_size)}")
        print(f"Quantized model size (disk):  {format_bytes(quant_size)}")
        print(f"Compression ratio:            {ratio:.2f} : 1")
        print(f"Storage saved:                {(1 - 1/ratio)*100:.1f}%")
    else:
        print(f"Quantized model total size:    {format_bytes(quant_size)}")
    print("="*50)
    print("Done.")