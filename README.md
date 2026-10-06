# AI/ML Systems: From Model to Infrastructure

A hands-on AI/ML systems learning repository built progressively from a simple PyTorch model toward GPU execution, profiling, inference performance, model serving, LLM systems, multi-GPU execution, and distributed AI infrastructure.

The goal is to connect:

**Model behavior → PyTorch → GPU execution → performance → serving → distributed systems → AI infrastructure**

Rather than treating these as separate topics, each project extends the previous one with hands-on implementation and measurement.

---

# Project 01 — Server Failure Prediction with PyTorch

## Objective

Build an end-to-end binary classification workflow that predicts server failure probability from synthetic server telemetry.

## Dataset

The synthetic dataset contains:

- 10,000 server samples
- 5 input features per server
- 1 binary failure label per server
- 80/20 train-test split

Feature tensor:

`X.shape = [10000, 5]`

Label tensor:

`y.shape = [10000, 1]`

Training tensors:

`X_train.shape = [8000, 5]`

`y_train.shape = [8000, 1]`

Test tensors:

`X_test.shape = [2000, 5]`

`y_test.shape = [2000, 1]`

## ML Workflow

The implementation covers:

**Data generation**

↓

**Feature normalization**

↓

**Tensor construction**

↓

**Train/test split**

↓

**Model construction**

↓

**Forward pass**

↓

**Logits**

↓

**BCEWithLogitsLoss**

↓

**Backpropagation**

↓

**SGD optimization**

↓

**Evaluation**

↓

**Inference**

## Model

A simple PyTorch linear binary classifier:

`nn.Linear(5, 1)`

The model learns:

- 5 weights
- 1 bias

## Evaluation Results

Test results from the completed experiment:

- Accuracy: **80.5%**
- Precision: **80.1%**
- Recall: **75.4%**
- F1: **77.7%**

Confusion matrix:

- True Negatives: **933**
- False Positives: **168**
- False Negatives: **221**
- True Positives: **678**

## Inference

The trained model was used to perform inference on a new server sample and generate a failure probability.

This completed the basic ML lifecycle:

**data → training → evaluation → inference**

### Notebook

`01_server_failure_prediction_pytorch.ipynb`

---

# Project 02 — GPU/CUDA Training and CPU vs GPU Benchmarking

Project 02 moves the ML workload from CPU execution to an NVIDIA GPU using PyTorch CUDA.

The objective was not simply to use a GPU, but to understand:

- where tensors live
- where model parameters live
- how computation moves from CPU to GPU
- when GPU execution actually improves performance

## GPU Environment

The experiment was executed in Google Colab using:

- PyTorch
- CUDA
- NVIDIA Tesla T4
- 1 GPU

CUDA availability was verified using:

`torch.cuda.is_available()`

## Tensor Device Placement

A tensor representing one server was first created on the CPU:

`shape = [1, 5]`

`device = cpu`

The same tensor was then moved to the GPU:

`shape = [1, 5]`

`device = cuda:0`

The shape did not change.

Only the execution device changed.

This demonstrates an important concept:

> Moving a tensor to a GPU changes where the tensor is stored and processed, not what the tensor represents.

## Training Data on GPU

The train/test tensors were moved to CUDA:

`X_train = [8000, 5] → cuda:0`

`y_train = [8000, 1] → cuda:0`

`X_test = [2000, 5] → cuda:0`

`y_test = [2000, 1] → cuda:0`

The model parameters were also moved to:

`cuda:0`

## GPU Training

The same fundamental ML training loop was executed on the GPU:

**Forward pass**

↓

**Loss**

↓

**Backpropagation**

↓

**Gradients**

↓

**SGD**

↓

**Weight update**

The training loss decreased from:

`0.7721 → 0.5850`

over 101 epochs.

The logits, loss calculation, model parameters, and evaluation tensors were verified on `cuda:0`.

---

## CPU vs GPU Benchmark

Two experiments demonstrated that GPU acceleration depends heavily on workload size and available parallelism.

### Small Workload

Training workload:

- 8,000 samples
- 5 features

Results:

- CPU: **0.0542 seconds**
- Tesla T4 GPU: **0.0745 seconds**

For this small workload, the CPU was faster.

The workload was too small to make effective use of the GPU, and GPU execution overhead outweighed the available parallel computation.

### Large Workload

The experiment was then scaled to:

- 1,000,000 samples
- 100 features
- 100,000,000 feature values

Results:

- CPU: **1.8489 seconds**
- Tesla T4 GPU: **0.1005 seconds**
- GPU speedup: **~18.4×**

## Key Finding

**GPU availability does not automatically mean faster execution.**

For small workloads, execution overhead can dominate.

As computational intensity and parallelism increase, GPU hardware can provide substantial acceleration.

The experiment demonstrated a transition from a workload where the CPU was faster to a larger workload where the NVIDIA Tesla T4 achieved approximately:

**18.4× faster measured training execution**

### Notebook

`02_gpu_cuda_training.ipynb`

---

# Project 03 — GPU Profiling and Bottleneck Analysis

Project 03 moves beyond measuring total execution time and begins answering a more important systems question:

> **Why is the workload fast or slow?**

The experiment uses system-level GPU monitoring and PyTorch operation-level profiling to identify where execution time is being spent.

---

## Step 1 — GPU System Observation

The GPU was first inspected using:

`nvidia-smi`

Before loading the workload:

- GPU: **Tesla T4**
- GPU memory usage: **3 MiB / 15,360 MiB**
- GPU utilization: **0%**
- Power: **10 W / 70 W**
- No active GPU process

This established the idle baseline.

---

## Step 2 — Load the GPU Workload

A larger workload was created:

`X.shape = [1,000,000, 100]`

`y.shape = [1,000,000, 1]`

Both tensors and the model were placed on:

`cuda:0`

PyTorch reported approximately:

**385.82 MB GPU memory allocated**

A second `nvidia-smi` observation showed:

- GPU memory usage: approximately **509 MiB**
- Python process using approximately **506 MiB**
- GPU utilization at the instant sampled: **0%**
- Power increased to approximately **32 W**

This demonstrated an important distinction:

> **GPU memory occupancy does not mean the GPU compute units are continuously busy.**

Data can remain resident in GPU memory while compute utilization is low or idle.

---

# PyTorch Profiler

The workload was then analyzed using:

`torch.profiler`

with both:

- `ProfilerActivity.CPU`
- `ProfilerActivity.CUDA`

This moved the analysis from:

**“How busy is the GPU?”**

to:

**“Which PyTorch operations are consuming GPU execution time?”**

---

## Profiler Findings

The dominant CUDA operations were:

### `aten::addmm`

Approximately:

**56% of self CUDA time**

This operation corresponds primarily to the linear layer's matrix operation plus bias.

### `aten::mm`

Approximately:

**35% of self CUDA time**

This represents matrix multiplication associated with the model's computation, including backward/gradient work.

Other elementwise and loss-related operations consumed substantially less CUDA time.

The profiler therefore showed that the dominant useful GPU computation was concentrated in the model's linear algebra operations.

---

# Batch-Size / Throughput Experiment

After identifying the dominant operations, the next experiment investigated how the amount of work presented in each training step affected GPU throughput.

The total dataset remained:

**1,000,000 samples**

The batch size was varied while processing one pass over the dataset.

## Results

| Batch Size | Time | Throughput |
|---:|---:|---:|
| 1,000 | 0.8632 sec | 1,158,480 samples/sec |
| 10,000 | 0.0850 sec | 11,770,085 samples/sec |
| 100,000 | 0.0089 sec | 112,449,712 samples/sec |
| 1,000,000 | 0.0047 sec | 213,344,166 samples/sec |

Measured throughput increased from:

**1.16 million samples/sec**

to:

**213.34 million samples/sec**

This represents approximately:

**184.2× higher measured processing throughput**

between the smallest and largest batch configurations in this experiment.

---

## Why Did Throughput Increase?

With a batch size of 1,000, processing one million samples required many repeated training steps:

**many batches**

↓

**many forward passes**

↓

**many backward passes**

↓

**many optimizer steps**

↓

**repeated Python/CUDA execution overhead**

With a batch size of 1,000,000:

**one large batch**

↓

**one forward pass**

↓

**one backward pass**

↓

**one optimizer step**

↓

**far more work exposed to the GPU per training step**

The experiment therefore showed that repeated small-batch execution overhead was a major performance factor for this synthetic workload.

---

## Important Interpretation

The **184.2× result is a systems-throughput measurement**, not a claim that the ML model itself became 184× better.

Changing batch size also changes the number of optimizer updates performed during one pass through the dataset.

In real model training, batch size can affect:

- convergence
- optimization behavior
- GPU memory consumption
- model quality
- training dynamics

This experiment intentionally focused on **GPU execution throughput and repeated training-step overhead**.

---

# Profiling After Scaling

The large-batch workload was profiled again.

The dominant CUDA operations remained approximately:

- `aten::addmm` → **~56% self CUDA time**
- `aten::mm` → **~35% self CUDA time**

This is expected.

The optimization experiment did not eliminate the model's useful matrix computation.

Instead, it increased the amount of useful work performed per training step and reduced repeated execution overhead across the dataset.

---

# Colab 3 Performance-Debugging Workflow

The completed experiment followed this process:

**Observe**

↓

`nvidia-smi`

↓

**Measure**

↓

PyTorch Profiler

↓

**Identify dominant operations**

↓

`addmm / mm`

↓

**Form hypothesis**

↓

small-batch execution overhead limits throughput

↓

**Change workload organization**

↓

increase batch size

↓

**Measure again**

↓

1.16M → 213.34M samples/sec

↓

**Explain the result**

This creates a repeatable performance-debugging methodology:

> **Observe → Profile → Identify → Change → Measure → Explain**

### Notebook

`03_gpu_profiling_bottleneck_analysis.ipynb`

---

# What I Have Implemented So Far

## ML Model Layer

- Synthetic dataset generation
- Feature normalization
- PyTorch tensor construction
- Train/test splitting
- Binary classification
- Model training
- Forward pass
- Loss computation
- Backpropagation
- SGD optimization
- Accuracy evaluation
- Precision
- Recall
- F1
- Confusion matrix
- Inference

## GPU Execution Layer

- CUDA availability detection
- CPU/GPU device placement
- Tensor movement to GPU
- Model movement to GPU
- GPU forward pass
- GPU loss computation
- GPU backpropagation
- GPU evaluation
- CUDA synchronization
- CPU vs GPU benchmarking
- Workload scaling
- Measured GPU acceleration

## GPU Performance Layer

- `nvidia-smi`
- GPU memory observation
- GPU utilization observation
- PyTorch Profiler
- CPU and CUDA activity profiling
- Operation-level CUDA timing
- Identification of dominant GPU operations
- Batch-size experimentation
- Throughput measurement
- Before/after performance comparison
- Bottleneck reasoning

---

# Roadmap

- [x] **01 — Basic PyTorch model:** data → training → evaluation → inference
- [x] **02 — GPU/CUDA training:** device placement → GPU training → CPU/GPU benchmarking
- [x] **03 — GPU profiling:** nvidia-smi → PyTorch Profiler → bottleneck analysis → throughput optimization
- [ ] **04 — Inference performance**
- [ ] **05 — Model serving**
- [ ] **06 — Transformer / LLM inference foundations**
- [ ] **07 — KV cache and vLLM**
- [ ] **08 — Multi-GPU execution**
- [ ] **09 — Distributed communication: NCCL / NVLink / high-performance networking**
- [ ] **10 — From GPU server to rack-scale AI infrastructure**

---

# Next — Inference Performance

Training asks:

> **How efficiently can we teach the model?**

Inference asks:

> **How efficiently can the trained model answer requests?**

The next project will investigate:

- inference latency
- throughput
- batch size
- CPU vs GPU inference
- warm-up behavior
- synchronization
- `torch.no_grad()`
- model evaluation mode
- request batching
- latency vs throughput trade-offs

This begins the transition from:

**model training**

to:

**production model serving**

---

# Long-Term Goal

Build a practical understanding of the complete AI systems path:

**Model**

↓

**PyTorch**

↓

**CUDA / GPU execution**

↓

**GPU profiling**

↓

**Performance optimization**

↓

**Inference**

↓

**Model serving**

↓

**Transformer / LLM systems**

↓

**KV cache / high-throughput inference**

↓

**Multi-GPU execution**

↓

**NCCL / NVLink / high-performance networking**

↓

**GPU server**

↓

**Rack-scale AI infrastructure**

Each stage adds hands-on implementation, measurement, debugging, and systems-level understanding.
