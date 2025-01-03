# Training Script Documentation

## Overview
This document describes the usage of the `train.sh` script for model training with DeepSpeed integration.

## Default Parameters
```bash
NNODES=1                      # Number of nodes
GPUS_PER_NODE=4              # GPUs per node
MICRO_BSZ=1                  # Micro batch size
ACCUMULATE_GRAD_BATCHES=4    # Gradient accumulation steps
MAX_LENGTH=512               # Maximum sequence length
LR_INIT=6e-4                 # Initial learning rate
LR_FINAL=1e-5               # Final learning rate
WARMUP_STEPS=50             # Warmup steps
GRAD_CP=1                   # Gradient checkpointing
DEEPSTATE_STAGE=3           # DeepSpeed stage
MAX_TRAINED_TOKENS=100M     # Maximum training tokens
TERMINATE_LOSS=0.01         # Termination loss threshold
```

## Script Options
| Option | Description |
|--------|-------------|
| -c | Config file path |
| -o | Output directory |
| -p | Preprocessed data path |
| -r | Raw data directory |
| -n | Number of nodes |
| -m | Maximum sequence length |
| -b | Micro batch size |
| -a | Accumulate gradient batches |
| -l | Initial learning rate |
| -f | Final learning rate |
| -w | Warmup steps |
| -k | Checkpoint file |
| -g | Gradient checkpointing (0/1) |
| -d | Enable DeepSpeed offload |
| -F | Enable full parameters |
| -s | Training stage (1/2) |
| -R | RWKV version |
| -W | WKV setting |
| -S | DeepSpeed stage |
| -t | Maximum trained tokens |
| -T | Terminate loss threshold |
| -P | Wandb project name |
| -G | GPUs per node |
| -M | Enable group norm |
| -z | Freeze MLP |
| -i | Teacher model ID |

## Training Examples

### 1. Stage 1 Training (Qwen 0.5B with Norm)
```bash
sh train.sh \
  -c configs/qwen_0.5b.yaml \
  -l 0.0001 \
  -f 0.00001 \
  -m 2048 \
  -b 2 \
  -r "/home/yueyulin/data/finemath/finemath-4plus/ /home/yueyulin/data/Mobius/standard/ /home/yueyulin/data/dclm-10B/ /home/yueyulin/data/additional_jsonl_cut/" \
  -o /home/yueyulin/model/qwen_0.5b_full_layers_stage1_v7_finemath \
  -g 1 \
  -F 0 \
  -d 1 \
  -t 1000_000_000 \
  -T 0.2 \
  -R v7 \
  -s 1 \
  -M 1
```

### 2. Stage 2 Training (Qwen 0.5B with Norm)
```bash
sh train.sh \
  -c configs/qwen_0.5b.yaml \
  -l 0.0001 \
  -f 0.00001 \
  -m 2048 \
  -b 2 \
  -r "/home/yueyulin/data/Magpie-Qwen2.5-Pro-1M-v0.1/data /home/yueyulin/data/finemath/finemath-4plus/ /home/yueyulin/data/Mobius/standard/ /home/yueyulin/data/dclm-10B/ /home/yueyulin/data/additional_jsonl_cut/" \
  -o /home/yueyulin/model/qwen_0.5b_stage2_v7_finemath \
  -g 1 \
  -F 0 \
  -d 1 \
  -t 1000_000_000 \
  -T 0.2 \
  -R v7 \
  -s 2 \
  -k /home/yueyulin/model/qwen_0.5b_full_layers_stage2_v7_finemath/pytorch_model.bin \
  -M 1
```

### 3. Stage 2 Training (Qwen 0.5B with Norm and Frozen MLP)
```bash
sh train.sh \
  -c configs/qwen_0.5b.yaml \
  -l 0.0001 \
  -f 0.00001 \
  -m 2048 \
  -b 2 \
  -r "/home/yueyulin/data/Magpie-Qwen2.5-Pro-1M-v0.1/data /home/yueyulin/data/finemath/finemath-4plus/ /home/yueyulin/data/Mobius/standard/ /home/yueyulin/data/dclm-10B/ /home/yueyulin/data/additional_jsonl_cut/" \
  -o /home/yueyulin/model/qwen_0.5b_stage2_v7_finemath \
  -g 1 \
  -F 0 \
  -d 1 \
  -t 1000_000_000 \
  -T 0.2 \
  -R v7 \
  -s 2 \
  -k /home/yueyulin/model/qwen_0.5b_full_layers_stage2_v7_finemath/pytorch_model.bin \
  -M 1 \
  -z 1
```

### 4. Stage 2 Training (Qwen 0.5B with Norm, Frozen MLP, and Custom Teacher Model)
```bash
sh train.sh \
  -c configs/qwen_0.5b.yaml \
  -l 0.0001 \
  -f 0.00001 \
  -m 2048 \
  -b 2 \
  -r "/home/yueyulin/data/Magpie-Qwen2.5-Pro-1M-v0.1/data /home/yueyulin/data/finemath/finemath-4plus/ /home/yueyulin/data/Mobius/standard/ /home/yueyulin/data/dclm-10B/ /home/yueyulin/data/additional_jsonl_cut/" \
  -o /home/yueyulin/model/qwen_0.5b_stage2_v7_finemath \
  -g 1 \
  -F 0 \
  -d 1 \
  -t 1000_000_000 \
  -T 0.2 \
  -R v7 \
  -s 2 \
  -k /home/yueyulin/model/qwen_0.5b_full_layers_stage2_v7_finemath/pytorch_model.bin \
  -M 1 \
  -z 1 \
  -i /home/yueyulin/models/Qwen2.5-7B-Instruct/
```

## Important Notes
1. The total batch size is calculated as: `NNODES * GPUS_PER_NODE * MICRO_BSZ * ACCUMULATE_GRAD_BATCHES`
2. World size is calculated as: `NNODES * GPUS_PER_NODE`
3. The script uses DeepSpeed for distributed training
4. Default save checkpoint interval is 2000 batches
5. Dropout is set to 0.05 by default