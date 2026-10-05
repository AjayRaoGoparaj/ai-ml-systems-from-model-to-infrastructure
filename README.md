# AI/ML Systems: From Model to Infrastructure

A hands-on learning repository that builds an AI/ML system step by step — starting with a simple PyTorch model and progressively extending the same mental model toward GPU execution, profiling, inference, serving, LLM systems, distributed communication, and rack-scale AI infrastructure.

The goal is to connect **model-level behavior** with the **GPU and infrastructure systems that execute AI workloads in production**.

---

## Project 01 — Server Failure Prediction with PyTorch

### Objective

Build an end-to-end binary classification workflow that predicts server failure probability from synthetic server telemetry.

### Dataset

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

### ML Workflow

The implementation covers:

Data generation  
→ feature normalization  
→ tensor construction  
→ train/test split  
→ model construction  
→ forward pass  
→ logits  
→ BCEWithLogitsLoss  
→ backpropagation  
→ SGD optimization  
→ evaluation  
→ inference

### Model

A simple PyTorch linear binary classifier:

`nn.Linear(5, 1)`

The model learns:

- 5 weights
- 1 bias

### Evaluation Results

Test results from the completed experiment:

- Accuracy: **80.5%**
- Precision: **80.1%**
- Recall: **75.4%**
- F1: **77.7%**

Confusion matrix:

- True Negatives: 933
- False Positives: 168
- False Negatives: 221
- True Positives: 678

### Inference

The trained model was used to perform inference on a new server sample and produce a failure probability.

This completes the basic model lifecycle:

**data → training → evaluation → inference**

Notebook:

`01_server_failure_prediction_pytorch.ipynb`

---

## Project 02 — GPU/CUDA Training and CPU vs GPU Benchmarking

The same ML workflow was extended from CPU execution to an NVIDIA GPU using PyTorch CUDA.

### GPU Environment

The experiment was executed in Google Colab using:

- PyTorch
- CUDA
- NVIDIA Tesla T4
- 1 GPU

CUDA availability was verified using:

`torch.cuda.is_available()`

### Tensor and Model Device Placement

The experiment demonstrated explicit device placement using:

`tensor.to(device)`

and:

`model.to(device)`

A tensor representing one server retained the same shape when moved between devices:

**CPU**

`shape = [1, 5]`  
`device = cpu`

**GPU**

`shape = [1, 5]`  
`device = cuda:0`

This demonstrates an important concept:

> Moving a tensor to the GPU changes where the tensor is stored and processed, not its shape or meaning.

The training and test tensors were then moved to the GPU:

`X_train = [8000, 5] → cuda:0`

`y_train = [8000, 1] → cuda:0`

`X_test = [2000, 5] → cuda:0`

`y_test = [2000, 1] → cuda:0`

The model parameters were also verified on `cuda:0`.

### GPU Training

The same fundamental training loop was executed on the GPU:

**forward pass → loss → backpropagation → gradients → SGD → weight update**

The training loss decreased from:

`0.7721 → 0.5850`

over 101 epochs.

The resulting logits, loss calculation, model parameters, and evaluation tensors were verified on `cuda:0`.

### CPU vs GPU Benchmark

Two controlled experiments demonstrated that GPU acceleration depends on workload size and computational parallelism.

#### Small workload

Training workload:

- 8,000 samples
- 5 features

Results:

- CPU: **0.0542 seconds**
- Tesla T4 GPU: **0.0745 seconds**

For this small workload, the CPU was faster because GPU execution overhead outweighed the small amount of parallel computation.

#### Large workload

The experiment was then scaled to:

- 1,000,000 samples
- 100 features
- 100,000,000 feature values

Results:

- CPU: **1.8489 seconds**
- Tesla T4 GPU: **0.1005 seconds**
- GPU speedup: **~18.4×**

### Key Finding

**GPU availability does not automatically mean faster execution.**

For small workloads, GPU overhead can dominate.

As computational intensity and parallelism increase, GPUs can provide substantial acceleration.

This experiment demonstrated a transition from a workload where the CPU was faster to a larger workload where the NVIDIA T4 GPU achieved approximately **18.4× faster training execution**.

Notebook:

`02_gpu_cuda_training.ipynb`

---

## What I Have Implemented So Far

### Model Layer

- Synthetic dataset generation
- Feature normalization
- PyTorch tensor construction
- Binary classification
- Train/test splitting
- Model training
- Loss computation
- Backpropagation
- SGD optimization
- Accuracy, precision, recall, and F1 evaluation
- Inference

### GPU Execution Layer

- CUDA availability detection
- CPU vs GPU device placement
- Tensor movement to GPU
- Model movement to GPU
- GPU-based forward pass
- GPU loss computation
- GPU backpropagation
- GPU evaluation
- CUDA synchronization for benchmarking
- CPU vs GPU performance measurement
- Workload scaling
- Measured GPU acceleration

---

## Roadmap

- [x] **01 — Basic PyTorch model:** data → training → evaluation → inference
- [x] **02 — GPU/CUDA training:** device placement → GPU training → evaluation → CPU/GPU benchmarking
- [ ] **03 — GPU profiling and bottleneck analysis**
- [ ] **04 — Inference performance**
- [ ] **05 — Model serving**
- [ ] **06 — Transformer / LLM inference foundations**
- [ ] **07 — KV cache and vLLM**
- [ ] **08 — Multi-GPU execution**
- [ ] **09 — Distributed communication: NCCL / NVLink / high-performance networking**
- [ ] **10 — From GPU server to rack-scale AI infrastructure**

---

## Next — GPU Profiling and Bottleneck Analysis

The next stage moves beyond measuring total runtime and asks:

**Why is the workload fast or slow?**

The next experiment will examine GPU execution using profiling and utilization tools to understand:

- GPU utilization
- CPU vs GPU bottlenecks
- kernel execution
- memory behavior
- workload size
- synchronization overhead
- compute utilization
- opportunities for optimization

The objective is to move from:

**“The GPU was faster.”**

to:

**“I measured where execution time was spent, identified the dominant bottleneck, changed the workload or implementation, and quantified the performance improvement.”**

---

## Long-Term Goal

Build a practical understanding of the complete AI systems path:

**Model**

↓  

**PyTorch**

↓  

**CUDA / GPU execution**

↓  

**Profiling and optimization**

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

The repository is intentionally built incrementally, with each stage adding hands-on implementation, measurements, and systems-level understanding.
