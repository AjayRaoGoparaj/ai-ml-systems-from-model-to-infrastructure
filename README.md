# AI/ML Systems: From Model to Infrastructure

A hands-on learning repository that builds one simple machine-learning problem step by step, then extends the same mental model toward GPU execution, inference, serving, and AI infrastructure.

## Project 01 — Server Failure Prediction with PyTorch

### Problem
Build a simple binary-classification model that predicts whether a server will fail in the next 24 hours.

This project uses **synthetic data for learning purposes**. It does not use production telemetry from AWS, Microsoft, or any employer.

### Features
Each synthetic server has five input features:

1. Temperature
2. Power
3. Memory errors
4. Fan speed
5. Age

The target label is:

- `0` — no failure
- `1` — failure

### End-to-end flow

```text
Synthetic server data
        ↓
5 input features + binary label
        ↓
PyTorch tensors
X [10000, 5]
y [10000, 1]
        ↓
80/20 train-test split
        ↓
nn.Linear(5, 1)
        ↓
Forward pass → logits
        ↓
BCEWithLogitsLoss
        ↓
Backpropagation → gradients
        ↓
SGD optimizer → updated weights
        ↓
Repeat for 101 epochs
        ↓
Trained model
        ↓
Evaluation on unseen test data
        ↓
Inference on one new server
```

### Observed run

In the completed Colab run:

| Metric | Result |
|---|---:|
| Initial training loss | 0.8960 |
| Training loss at epoch 100 | 0.4401 |
| Test accuracy | 0.805 |
| Precision | 0.801 |
| Recall | 0.754 |
| F1 | 0.777 |

Confusion-matrix counts:

- True negatives: 933
- False positives: 168
- False negatives: 221
- True positives: 678

For one illustrative new server with raw values `[92, 940, 42, 6400, 4]`, the trained synthetic model produced a failure probability of approximately **0.921**.

Because the dataset is synthetic, these values demonstrate the ML workflow rather than real-world server reliability.

### What this project demonstrates

- Creating and normalizing numerical features
- Understanding tensor shapes (`[N,F]` and `[N,1]`)
- Random train/test splitting
- Building a PyTorch `nn.Linear` binary classifier
- Understanding weights and bias
- Forward passes and logits
- `BCEWithLogitsLoss`
- Backpropagation and gradients
- SGD parameter updates
- Training-loss convergence
- Accuracy, precision, recall, F1, and confusion-matrix interpretation
- Evaluation with fixed model weights
- Inference on a new example

## Run it

Open `01_server_failure_prediction_pytorch.ipynb` in Google Colab and run the cells from top to bottom.

The first implementation intentionally runs on CPU. A later project will move the same workload to GPU so the CPU/GPU differences are explicit rather than hidden.

## Roadmap

- [x] 01 — Basic PyTorch model: data → training → evaluation → inference
- [ ] 02 — GPU training: tensors/model on CUDA
- [ ] 03 — GPU profiling and bottleneck analysis
- [ ] 04 — Inference performance
- [ ] 05 — Model serving
- [ ] 06 — Transformer/LLM inference foundations
- [ ] 07 — KV cache and vLLM
- [ ] 08 — Multi-GPU execution
- [ ] 09 — Distributed communication: NCCL / NVLink / high-performance networking
- [ ] 10 — From GPU server to rack-scale AI infrastructure

## Why this repository exists

The goal is to connect model-level ML mechanics to the infrastructure that eventually runs large AI workloads. Each project extends the same end-to-end systems view instead of treating ML, GPU performance, serving, and infrastructure as disconnected topics.
