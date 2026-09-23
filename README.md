# RLVR Reward Landscape

This repository provides code for visualizing the **reward landscape** of RLVR-trained language models.

We currently focus exclusively on **MATH500**. The visualization evaluates the model’s average reward along three types of parameter-space directions:

- **Random direction**: a randomly sampled direction in parameter space.
- **SGD direction**: a direction derived from stochastic gradient descent.
- **Displacement between adjacent steps**: the parameter difference between two consecutive training steps.

## Data

The dataset is located in `/train_data/math500`:

- `test/test_500.jsonl`: the complete set of 500 examples.
- `train.jsonl`: a training subset of 300 examples.
- `test.jsonl`: a test subset of 200 examples.

## Experimental Setup and Goal

Our initial experiments trained models on the 300-example training subset and visualized their reward landscapes on the same examples.

Our next focus is to repeatedly train on a fixed subset of **1 or 8 examples** until the model overfits, then visualize the reward landscape on those same examples.

The goal is to observe how the reward landscape differs across **training stages** and between **better- and worse-performing models**, including how it changes as training progresses toward overfitting.

## Usage

### Random Direction

```bash
bash scripts/run_qwen_math500_vllm.sh
```

### SGD Direction

```bash
bash scripts/run_qwen_math500_sgd.sh
```

### Displacement Between Adjacent Steps

```bash
bash scripts/run_qwen_math500_displacement.sh
```

The shell scripts contain the configurations for model checkpoints, dataset paths, and visualization hyperparameters. Please modify these configurations as needed before running.