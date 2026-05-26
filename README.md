LLIPSE: Lightweight LLM-backed Prediction Model for Inland Waterway Ship-to-Shore End-to-End Communications
=
Dynamic environmental factors, including seasonal weather variations, signal obstructions caused by bridges, and waterway geometry, 
significantly influence end-to-end (E2E) latency prediction in wireless ship-to-shore communications.
To address these challenges, this work presents a Multimodal Environmental Augmented Navigation Dataset (MEAND). 
MEAND integrates heterogeneous multimodal information, including meteorological variables, e.g., wind speed, temperature, and rainfall, 
detailed waterway geometry, and fixed geographic features such as bridges and small islands, all of which directly affect wireless transmission performance.

Dataset Construction
--
<img width="1213" height="663" alt="64e2d5c4bcfc710033a9ce8e6afa2e8" src="https://github.com/user-attachments/assets/7d1388ef-6a5a-4269-8be9-03dc2f2fe824" />

<img width="9160" height="4161" alt="fig2_01" src="https://github.com/user-attachments/assets/64bed185-5389-4bb9-973f-4d07d20bb868" />

MEAND expands FND by incorporating multimodal environmental information. In particular, inland waterways wind speed, temperature, and rainfall, are considered, as they directly influence wireless data transmission quality. Moreover, we integrate geographic information from the Yangtze waterway bureau, and marked fixed features, such as bridge, waterway geometry, and small islands in inland waterway.



## Requirements
- python ==3.8
- torch >= 1.8.0
- torchvision >= 0.9.0
- six
  
## Model Preparation

- [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct)
- [Deepseek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)

### Data preparation

```text
│MEAND/
├──train_labeled.json
│   ├── [0]                       # first conversation sample
│   │       └── conversations
│   │           ├── [0]               # query
│   │           │   ├── from: "human"
│   │           │   └── value: "Current time is night at 04:47 ..."
│   │           └── [1]               # gpt response
│   │               ├── from: "gpt"
│   │               └── value: "Predicted communication performance: Average delay is 84.2 ms, packet loss rate is 0.87%."
│   ├── [1]                           # second conversation sample
│   │   └── conversations
│   │       ├── [0]
│   │       │   ├── from: "human"
│   │       │   └── value: "What's the link performance looking like? ..."
│   │       └── [1]
│   │           ├── from: "gpt"
│   │           └── value: "Predicted communication performance: Average delay is 84.2 ms, packet loss rate is 0.87%."
│   └── ...
│
├──test_labeled.json
│    ├──[0] 
│    │    └── ...
│    └── ...
``` 

Model  Enhancement
--
We employ a foundation LLM as the backbone and adapt it for inland waterway communication performance prediction via EIB-LoRA. The adapted model is then compressed via quantization to enable lightweight inference on resource-constrained shipborne edge devices.

Frequently Asked Questions
--
Q1：The dataset appears relatively small. How do you ensure the model's performance isn't due to overfitting on limited data?<br>
A1：Collecting large-scale, fine-grained measurement-based data for inland waterway communications is inherently challenging due to practical constraints, such as limited infrastructure, complex waterway geometry, seasonal effects, and dynamic environmental conditions. These factors explain why datasets in this domain are typically limited in size. Notably, the data collected in this work are aggregated hourly, with each point representing a statistical characteristic (e.g., average or weighted latency) rather than a single-time instantaneous measurement. This aggregation reduces random noise and short-term fluctuations, thereby enhancing information accuracy and reliability. Moreover, to mitigate overfitting, we integrate Dropout regularization into our neural network architecture [1]. During training, this technique randomly deactivates a subset of neurons in each layer, encouraging the model to learn robust and generalizable features.


Q2: The networking aspects and wireless technology details are not clearly described. What specific communication technologies are utilized?<br>
A2: The system employs a heterogeneous networking architecture. In near-shore and inland waterway scenarios, 4G/5G cellular networks can support high data rate services, while the QUIC protocol ensures efficient, reliable, and low-latency data transmission over these wireless channels. AIS regularly broadcasts essential navigation information, and in some situations, satellite uses RTK positioning to provide high-precision positioning and time synchronization.


[1] Y. Lin, X. Ma, X. Chu, Y. Jin, Z. Yang, Y. Wang, and H. Mei, “LoRA Dropout as a sparsity regularizer for overfitting control,” arXiv preprint arXiv:2404.09610, 2024.

Repository Layout
--
```
.
├── data/                              # MEAND dataset
│   ├── Summer_InlandWaterwayComDataset.zip
│   ├── train_labeled.json
│   └── test_labeled.json
├── scripts/                           # Training & quantization code
│   ├── admm.py                        # ADMM 4-bit quantization
|   ├── train.py 
│   └── test.py                   

├── docs/                              # Supplementary material
│   ├── PIPELINE.md                    # End-to-end pipeline notes
│   ├── QApair.pdf
│   └── honghu_bailuo.fig
├── LICENSE                            # MIT — covers source code
└── DATA_LICENSE                       # CC BY-NC 4.0 — covers the dataset
```

License
--
This repository uses a dual-license scheme:

- **Source code** (everything under `scripts/`) is released under the
  **MIT License** — see [`LICENSE`](LICENSE).
- **Dataset and supplementary material** (everything under `data/` and
  `docs/`) is released under **Creative Commons Attribution-NonCommercial
  4.0 International (CC BY-NC 4.0)** — see [`DATA_LICENSE`](DATA_LICENSE).

When citing the dataset, please refer to the attribution block at the bottom
of [`DATA_LICENSE`](DATA_LICENSE).



