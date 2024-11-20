# Hybrid Fine-Tuning with Prompt Tuning and LoRA on FLAN-T5

This project focuses on enhancing parameter efficiency in fine-tuning large language models by integrating **Prompt Tuning** and **Low-rank Adaptation (LoRA)**. The approach is evaluated on diverse NLP tasks, including SQL query generation, dialogue summarization, and sentiment analysis.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Project Overview

This repository implements a **hybrid fine-tuning strategy** for the FLAN-T5 model, combining:
- **Prompt Tuning**: Soft tokens for task-specific guidance.
- **LoRA**: Efficient structural adaptations to transformer layers.

The approach is designed to:
1. Achieve competitive performance with minimal trainable parameters.
2. Scale fine-tuning efficiently for diverse tasks.
3. Enhance model adaptability to unseen tasks.

## Folder Structure

```
Hybrid-PEFT/
│
├── configs/                                # Configuration files for training
│   ├── train_config_<dataset>_lora_pt.json # Config for for hybrid fine-tuning
│   ├── train_config_<dataset>_lora.json    # Config for LoRA training
│   ├── train_config_<dataset>_pt.json      # Config for Prompt Tuning
│   ├── train_config_<dataset>.json         # Config for full fine-tuning
│
├── model_logs/                             # Directory for model checkpoints and logs
│
├── notebooks/                              # Jupyter notebooks for data exploration and analysis
│   ├── 01-look-at-data.ipynb               # Data exploration and visualization
│   ├── 02-few-shot-inference.ipynb         # Few-shot inference
│   ├── 03-look-at-model-param-count.ipynb  # Parameter analysis
│   ├── 04-look-at-prepared-data.ipynb      # Prepared data and output analysis
│   ├── 05-look-at-attributions.ipynb       # Attribution analysis
│
├── report/                                 # Project report and documentation
│   ├── Hybrid LoRA and Prompt Tuning.pdf   # Detailed project report
│
├── scripts/                                # Shell scripts for running experiments
│   ├── eval_hybrid_peft.sh                 # Script for evaluation
│   ├── train_hybrid_peft.sh                # Script for training
│
├── src/                                    # Source code for data processing, training, and evaluation
│   ├── data_preparation.py                 # Functions for preparing datasets
│   ├── metrics.py                          # Custom metrics for evaluation
│   ├── ml_pipeline.py                      # Training and evaluation pipeline
│   ├── peft_wrappers.py                    # LoRA and Prompt Tuning integration
│   ├── training.py                         # Model training script
│   ├── arg_parse.py                        # Argument parsing
│   ├── utils.py                            # Helper functions
│   ├── task_adapter.py                     # Task-specific adapter
│   ├── attributions.py                     # Model interpretability utilities
│
├── requirements.txt                        # Python dependencies
├── eval_hybrid_peft.py                     # Python script for model evaluation
└── train_hybrid_peft.py                    # Python script for model training
```

## Setup and Installation

### Requirements
- Python 3.8+
- PyTorch (with CUDA support)
- Hugging Face Transformers
- `datasets` and `accelerate` for data handling and distributed training

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shahinntu/Hybrid-LoRA-and-Prompt-Tuning.git
   cd Hybrid-LoRA-and-Prompt-Tuning
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the FLAN-T5 model using the hybrid approach, execute the training script with the appropriate configuration file included in the `.sh` file:

```bash
bash scripts/train_hybrid_peft.sh
```

### Evaluation

To evaluate the trained model, execute the evaluation script with the appropriate configuration file included in the `.sh` file:

```bash
bash scripts/eval_hybrid_peft.sh
```

Make sure the `.sh` files are updated with the correct paths to the configuration files for your dataset and fine-tuning method.

### Jupyter Notebooks

Explore the data and analyze model behavior using the notebooks in the `notebooks/` directory.

## Configuration

Training configurations can be customized using the JSON files in the `configs/` directory:
- **Batch Size**: Number of samples processed per batch.
- **Learning Rate**: Adjust the model's learning rate for optimal training.
- **LoRA Parameters**: Configure rank, alpha, and target layers.
- **Prompt Tuning Parameters**: Modify the number of soft tokens.

Edit these files to suit your experimental setup.
