End-to-end pipeline: 4‑bit ADMM quantization → LoRA fine‑tuning with EIB strategy → evaluation for conversation latency prediction.

## Files

- `admm.py` – offline layer‑wise ADMM 4‑bit quantization (saves int8 packed weights)
- `train.py` – loads quantized backbone, injects LoRA + IB gates, trains regression head (MSE + gate penalty)
- `test.py` – merges LoRA into quantized weights, evaluates on test set (MAE, RMSE, MAPE, inference time)

## Quick Start

### 1. Set paths in each script

| Script | Key variables |
|--------|----------------|
| `admm_4bit_quantize.py` | `MODEL_PATH`, `QUANT_OUTPUT_DIR` |
| `train_ib_lora.py` | `ADMM_QUANTIZED_PATH`, `FULL_MODEL_PATH`, `TRAIN_DATA_PATH`, `TRAINING_OUTPUT_DIR` |
| `test_final_model.py` | `ADMM_QUANTIZED_PATH`, `FULL_MODEL_PATH`, `TRAINING_OUTPUT_DIR`, `TEST_DATA_PATH` |

  
Data Format
### 2. Run quantization
{
  "conversations": [
    {"from": "human", "value": "user message"},
    {"from": "gpt", "value": "response ... Average delay is 123.45 ms"}
  ]
}
## License

This project is released under the **MIT License**.  
See the [LICENSE](LICENSE) file in the repository root for full text.
