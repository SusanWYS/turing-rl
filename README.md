# Turing-RL

This repository contains code for the paper "Learning User Simulators with Turing Rewards".

## Repository Structure

```text
turing-rl/
├── bash_scripts/       # Reproducible launch wrappers
│   ├── data/           # Data + SFT-data generation
│   ├── sft/            # LoRA SFT launcher
│   ├── grpo/           # GRPO launcher
│   └── eval/           # Heldout generation + scoring
├── data/               # Dataset builders and splitters
│   ├── convokit/
│   ├── prism/
│   └── sft/
├── training/           # Trainable model workflows
│   ├── sft/
│   └── grpo/
├── eval/               # Heldout generation/scoring CLIs
└── shared/             # Shared prompts, judge utilities, model IDs, env loading
```

## Quickstart

### Install

```bash
python -m pip install -r requirements.txt
```

### Configure API keys

Create a `.env` file in the repository root:

```bash
OPENROUTER_API_KEY=...
OPENAI_API_BASE=https://openrouter.ai/api/v1
HF_TOKEN=...
WANDB_API_KEY=...
```

### Step 1: Generate Data

```bash
bash_scripts/data/generate_data.sh <data> <persona_inductor>
bash_scripts/data/generate_sft_data.sh <data> <persona_inductor>
```

### Step 2: SFT Warm Start

```bash
bash_scripts/sft/train_sft.sh <data> <condition> <persona_inductor>
```

### Step 3: GRPO Training

```bash
bash_scripts/grpo/train_grpo.sh <reward> <data> <condition> <persona_inductor>
```

GRPO starts from the matching SFT adapter. To override it:

```bash
SFT_ADAPTER_PATH=<adapter_dir> bash_scripts/grpo/train_grpo.sh <reward> <data> <condition> <persona_inductor>
```

### Step 4: Heldout Eval

```bash
bash_scripts/eval/generate_test.sh <model> <data> <condition> <train_mode> <persona_inductor>
bash_scripts/eval/score_test.sh <model> <data> <condition> <train_mode> <persona_inductor> <metric>
```

For trained checkpoints, `generate_test.sh` infers the checkpoint directory from the arguments. To override it:

```bash
CHECKPOINT_DIR=<checkpoint_dir> bash_scripts/eval/generate_test.sh <model> <data> <condition> <train_mode> <persona_inductor>
```

### Supported options

| Argument | Values | Meaning |
|---|---|---|
| `data` | `convokit`, `prism` | Domain: Reddit (`convokit`) or chat (`prism`). |
| `condition` | `history`, `persona`, `history_persona` | User representation conditions. |
| `persona_inductor` | `gpt-5.4-nano`, `opus4.8`, `qwen3-8b` | Model used to induce persona. |
| `reward` | `turing`, `sim`, `logprob` | GRPO reward family. |
| `model` | `qwen3-8b`, `qwen3.5-397b`, `gpt-5` | Heldout generation model. |
| `train_mode` | `none`, `sft`, `turing`, `sim`, `logprob` | Which trained checkpoint family to evaluate. |
| `metric` | `turing`, `sim`, `specificity`, `all` | Heldout scoring metric. |
| `SFT_ADAPTER_PATH` | path | If provided, override the inferred SFT adapter used to initialize GRPO. |
| `CHECKPOINT_DIR` | path | If provided, override the inferred trained-checkpoint directory. |

Use `train_mode=none` for untrained baselines.
