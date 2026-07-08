# Cassava Leaf Disease Classification Pipeline

A PyTorch-based machine learning pipeline designed for the [Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification) task. This repository provides a robust, scalable framework for training deep learning models, utilizing distributed training, mixed precision, and strict configuration management.

## System Architecture and Features

* **Distributed Training:** Integrated `DistributedDataParallel` (DDP) for scalable multi-GPU model training.
* **Mixed Precision (AMP):** Implementation of `torch.amp.autocast` and `GradScaler` to optimize memory utilization and computational throughput.
* **Configuration Management:** Strict hyperparameter validation and serialization utilizing Pydantic and YAML configuration files.
* **Dependency Management:** Built and locked using `uv` for deterministic builds and rapid environment resolution (`pyproject.toml` / `uv.lock`).
* **Experiment Tracking:** Native integration with [Weights & Biases (WandB)](https://wandb.ai/petrosangosa2005-ssss/VisionShift-Cassava/reports/VisionShift-Cassava-performance-narrative-and-next-training-step--VmlldzoxNzQ0NTI2NA?accessToken=jcyogm240awqv56gwsm4sfs0vgq57m929phyi1ukw2ojfes1w6y3uugmq2vd8uas) for logging metrics, hardware utilization, and model checkpoints.
* **Testing and CI/CD:** Unit testing implemented via `pytest` with automated workflows configured in GitHub Actions.

## Project Structure

```text
.
├── .github/workflows/      # CI/CD pipeline configurations
├── configs/                # YAML configuration files (baseline.yaml, kaggle.yaml)
├── data/                   # Ignored by Git (see Dataset section below)
│   ├── processed/          
│   └── raw/                
├── notebook/               # Exploratory Data Analysis (EDA) notebooks
├── src/                    # Core source code modules
│   ├── config.py           # Pydantic schema definitions
│   ├── data.py             # PyTorch Dataset and DataLoader implementations
│   ├── distributed.py      # Multi-GPU synchronization utilities
│   ├── losses.py           # Custom loss functions (e.g., Focal Loss)
│   ├── metrics.py          # Metric computation logic
│   ├── models.py           # Neural network architectures
│   ├── trainer.py          # Core training and validation loops
│   └── utils.py            # Helper functions (EarlyStopping, logging)
├── tests/                  # Unit tests (test_config.py, test_config.py)
├── pyproject.toml          # Project metadata and dependency definitions
├── train.py                # Main execution script for training
└── uv.lock                 # Locked dependency tree


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

*Note: The `data/processed/` directory will be generated automatically at runtime if required by the pipeline.*

## Installation

This project strictly utilizes `uv` for Python environment and dependency management. Ensure `uv` is installed on your system prior to setup.

1. Clone the repository:

```bash
git clone https://github.com/ssssssshy/Cassava-Leaf-Disease-Classification.git
cd Cassava-Leaf-Disease-Classification


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

To execute the main training script:

```bash
uv run train.py


```

To execute specific modules within the source directory, use the `-m` flag:

```bash
uv run -m src.trainer
uv run -m src.data


```

For distributed training across multiple GPUs, utilize `torchrun` within the `uv` context. Example for a 2-GPU node:

```bash
uv run torchrun --nproc_per_node=2 train.py


```

## ⚠️ Known Nuance: Distributed Validation (Padding Bias)

When evaluating the model during distributed multi-GPU training, please note a specific behavior regarding PyTorch's `DistributedSampler`:

By default, `DistributedSampler` pads (duplicates) the validation dataset at the end so that its total length is evenly divisible by `world_size` (the number of GPUs).

**The Issue:** Because of this padding, a few validation samples will be evaluated more than once. This can introduce a slight distortion (bias) into the final validation metrics.

**The Solution for Strict Inference:** While this slight bias is generally acceptable for monitoring trends during the training loop, strict validation or final inference requires a workaround. To obtain exact metrics, you must either:

1. Gather all predictions across ranks (e.g., using `all_gather`) and explicitly truncate the final predictions array to the original dataset size (`len(val_dataset)`).
2. Utilize a custom sampler that does not apply padding.

Please keep this in mind when analyzing final validation scores or performing strict inference generation on distributed setups.

## Configuration

Model hyperparameters, paths, and training routines are controlled via YAML files located in the `configs/` directory. These configurations are parsed and strictly validated at runtime by Pydantic models defined in `src/config.py`. To modify training parameters (e.g., learning rate, batch size, target metric for EarlyStopping), edit the respective YAML file before initiating the run.

## Development and Testing

To execute the test suite:

```bash
uv run pytest tests/


```