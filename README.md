# RLVR Reward Landscape

This repository provides code for visualizing the **reward landscape** of RLVR-trained language models.

The visualization evaluates the average reward of a trained model along randomly sampled parameter-space directions. Currently, the repository supports two reasoning tasks:

- GSM8K
- Chess

The chess task related model and data are from the project https://arxiv.org/pdf/2607.16097. You can download the chess llm checkpoints from huggingface: https://huggingface.co/chess-pre-to-post. 
The checkpoints starting with "rl" are trained with GRPO. 
The RL training data is in train_data/chess_data.

## Usage

### GSM8K

```bash
bash scripts/run_qwen_gsm8k.sh
```

### Chess

```bash
bash scripts/run_qwen_chess.sh
```

The shell scripts contain all necessary configurations, including the model checkpoint, dataset path, and visualization hyperparameters. Please modify these paths as needed before running.
