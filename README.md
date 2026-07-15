# Cassava Leaf Disease Classification Pipeline

A PyTorch-based machine learning pipeline designed for the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) task. This repository provides a robust, scalable framework for training deep learning models, supporting both classic CNN architectures (via `timm`) and state-of-the-art YOLO classification models (via `ultralytics`). It utilizes distributed training, mixed precision, and strict configuration management.

## System Architecture and Features

* **Multi-Architecture Support:** Seamlessly switch between standard models (e.g., InceptionV3) and Ultralytics YOLO classifiers (e.g., YOLOv12) using a unified factory pattern in `src/models.py`.
* **Smart Transfer Learning:** Automatic backbone weight initialization for YOLO classification models from pretrained detection weights, accelerating convergence.
* **Distributed Training:** Integrated `DistributedDataParallel` (DDP) for scalable multi-GPU model training.
* **Mixed Precision (AMP):** Implementation of `torch.amp.autocast` and `GradScaler` to optimize memory utilization and computational throughput.
* **Configuration Management:** Strict hyperparameter validation and serialization utilizing Pydantic and YAML configuration files.
* **Dependency Management:** Built and locked using `uv` for deterministic builds and rapid environment resolution (`pyproject.toml` / `uv.lock`).
* **Experiment Tracking:** Native integration with [Weights & Biases (WandB)](https://wandb.ai/petrosangosa2005-ssss/VisionShift-Cassava/reports/VisionShift-Cassava-performance-narrative-and-next-training-step--VmlldzoxNzQ0NTI2NA?accessToken=jcyogm240awqv56gwsm4sfs0vgq57m929phyi1ukw2ojfes1w6y3uugmq2vd8uas) for logging metrics, hardware utilization, and model checkpoints.
* **Model Serving:** Production-ready FastAPI service supporting both PyTorch (.pth) and ONNX (.onnx) inference.
* **ONNX Optimization:** Dedicated export script to convert trained models to ONNX format for high-performance serving.
* **Containerization:** Docker support using `uv` for fast, reproducible deployments.
* **Testing and CI/CD:** Unit testing implemented via `pytest` with automated workflows configured in GitHub Actions.

## Project Structure

```text
.
├── .github/workflows/      # CI/CD pipeline configurations
├── configs/                # YAML configuration files (inception.yaml, yolo12.yaml)
├── data/                   # Ignored by Git (see Dataset section below)
│   ├── processed/          
│   └── raw/                
├── src/                    # Core source code modules
│   ├── api.py              # FastAPI implementation for model serving
│   ├── export.py           # ONNX export and validation script
│   ├── config.py           # Pydantic schema definitions
│   ├── data.py             # PyTorch Dataset and DataLoader implementations
│   ├── distributed.py      # Multi-GPU synchronization utilities
│   ├── models.py           # Neural network architectures
│   └── ...                 
├── Dockerfile              # Container definition for API deployment
├── Dockerfile.gpu          # GPU-accelerated container definition
├── train.py                # Main execution script for training
└── ...

```

## Dataset Preparation

The dataset used in this project is provided by Kaggle. Due to its large size, the entire `data/` directory and its nested subfolders are explicitly excluded from version control via `.gitignore`.

You must download and structure the dataset locally before initiating any training routines.

1. Download the dataset via the Kaggle API:

```bash
kaggle competitions download -c cassava-leaf-disease-classification

```

2. Extract the archive and organize the files strictly according to this directory tree:

```text
data/
└── raw/
    ├── train_images/
    ├── train.csv
    └── label_num_to_disease_map.json

```

*Note: The `data/processed/` directory will be generated automatically at runtime if required by the pipeline. If running directly on Kaggle, ensure the absolute paths to `/kaggle/input/...` are correctly set in your YAML configs.*

## Installation

This project strictly utilizes `uv` for Python environment and dependency management. To download the model weights, you also need **Git LFS** installed on your system.

1. Clone the repository:
```bash
# Make sure Git LFS is installed on your system
# On Ubuntu/Debian: sudo apt-get install git-lfs
# On macOS: brew install git-lfs

git clone [https://github.com/ssssssshy/Cassava-Leaf-Disease-Classification.git](https://github.com/ssssssshy/Cassava-Leaf-Disease-Classification.git)
cd Cassava-Leaf-Disease-Classification

# Initialize Git LFS and pull the actual model weights (ONNX files)
git lfs install
git lfs pull

```

2. Synchronize the environment and install dependencies:

```bash
uv sync

```

3. (Optional) Activate the virtual environment explicitly:

```bash
# On Linux/macOS
source .venv/bin/activate

# On Windows
.venv\Scripts\activate

```

*Note: Explicit activation is optional. You can execute scripts directly via `uv run <command>` without activating the environment.*

4. Authenticate with Weights & Biases (required for metric tracking):

```bash
uv run wandb login

```

## Execution

Scripts and modules should be executed within the `uv` environment to ensure correct dependency resolution.

You can easily switch between architectures by pointing the training script to the respective configuration file:

To execute the main training script with standard CNN (Inception):

```bash
uv run train.py --config configs/inception.yaml

```

To execute the training script with YOLOv12 classification:

```bash
uv run train.py --config configs/yolo12.yaml

```

### ONNX Export

To convert a trained PyTorch checkpoint to ONNX for optimized inference:

```bash
# Ensure the root directory is in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
uv run python src/export.py --config configs/inception.yaml

```

### Model Serving (FastAPI)

To launch the inference API locally:

```bash
uv run uvicorn src.api:app --host 0.0.0.0 --port 8000

```

The API will automatically detect and prioritize the `.onnx` version of the model if it exists in the weights directory.

### Docker Deployment

By default, the project uses a lightweight `python:3.12-slim` image (via the standard `Dockerfile`) to ensure rapid API startup on a CPU. This is convenient for local testing on machines without dedicated hardware acceleration.

**1. Build the image:**

```bash
docker build -t yolo-api .

```

**2. Run the container:**
*(Model weights are mounted via a volume to prevent bloating the container image size)*

```bash
docker run -d \
  -p 8000:8000 \
  -e MODEL_CONFIG="/app/configs/yolo12.yaml" \
  -v $(pwd)/weights:/app/weights \
  --name cv-pipeline \
  yolo-api

```

> **Note for Production (GPU):**
> For deployment on servers with hardware acceleration, a dedicated `Dockerfile.gpu` based on the `nvidia/cuda` image is provided. To build the GPU-enabled container, run:
> `docker build -f Dockerfile.gpu -t yolo-api-gpu .`

---

### API Endpoints & Usage

* **`GET /health`**: Returns the API status.
* **`POST /predict`**: Submit an image file (multipart/form-data) for classification.

---

For distributed training across multiple GPUs, utilize `torchrun` within the `uv` context. Example for a 2-GPU node training YOLO:

```bash
uv run torchrun --nproc_per_node=2 train.py --config configs/yolo12.yaml

```

## ⚠️ Known Nuance: Distributed Validation (Padding Bias)

When evaluating the model during distributed multi-GPU training, please note a specific behavior regarding PyTorch's `DistributedSampler`:

By default, `DistributedSampler` pads (duplicates) the validation dataset at the end so that its total length is evenly divisible by `world_size` (the number of GPUs).

**The Issue:** Because of this padding, a few validation samples will be evaluated more than once. This can introduce a slight distortion (bias) into the final validation metrics.

**The Solution for Strict Inference:** While this slight bias is generally acceptable for monitoring trends during the training loop (which our custom metrics computation handles gracefully via internal synchronization), strict inference requires a workaround. To obtain exact metrics, you must either:

1. Gather all predictions across ranks (e.g., using `all_gather`) and explicitly truncate the final predictions array to the original dataset size (`len(val_dataset)`).
2. Utilize a custom sampler that does not apply padding.

Please keep this in mind when analyzing final validation scores or performing strict inference generation on distributed setups.

## Configuration

Model hyperparameters, paths, and training routines are controlled via YAML files located in the `configs/` directory. These configurations are parsed and strictly validated at runtime by Pydantic models defined in `src/config.py`.

Each model architecture has its own dedicated configuration file to handle distinct input sizes, learning rates, and save paths. To modify training parameters (e.g., learning rate, batch size, target metric for EarlyStopping), edit the respective YAML file before initiating the run.

## Development and Testing

To execute the test suite:

```bash
uv run pytest tests/

```

