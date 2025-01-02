#!/bin/bash
NNODES=1
GPUS_PER_NODE=4
MICRO_BSZ=1
ACCUMULATE_GRAD_BATCHES=4
export RWKV_VERSION=v6
MAX_LENGTH=512
CONFIG_FILE=toys_playground/configs/qwen7B_KL_Local.yaml
OUTPUT_DIR=toys_playground/output
PREPROCESSED_DATA=""
RAW_DATA_DIR=""
LR_INIT=6e-4
LR_FINAL=1e-5
WARMUP_STEPS=50
CKPT_FILE=""
GRAD_CP=1
DEEPSPEED_OFFLOAD=""
FULL_PARAMS=""
STAGE=1
export WKV=""
DEEPSTATE_STAGE=3
MAX_TRAINED_TOKENS=100_000_000
TERMINATE_LOSS=0.01
WANDB=hybrid_trainer_toys
WANDB_PROJECT=hybrid_trainer_toys
HAS_GROUP_NOMR=""
while getopts "c:o:p:n:m:b:a:l:f:w:k:g:d:F:s:R:W:S:t:T:W:P:r:G:M:" opt; do
    case $opt in
        c) CONFIG_FILE="$OPTARG";;
        o) OUTPUT_DIR="$OPTARG";;
        p) PREPROCESSED_DATA="--preprocessed_data $OPTARG";;
        r) RAW_DATA_DIR="--raw_data $OPTARG";;
        n) NNODES="$OPTARG";;
        m) MAX_LENGTH="$OPTARG";;
        b) MICRO_BSZ="$OPTARG";;
        a) ACCUMULATE_GRAD_BATCHES="$OPTARG";;
        l) LR_INIT="$OPTARG";;
        f) LR_FINAL="$OPTARG";;
        w) WARMUP_STEPS="$OPTARG";;
        k) CKPT_FILE="--ckpt_file $OPTARG";;
        g) GRAD_CP="$OPTARG";;
        d) DEEPSPEED_OFFLOAD="--deepspeed_offload";;
        F) FULL_PARAMS="--full_params";;
        s) STAGE="$OPTARG";;
        R) export RWKV_VERSION="$OPTARG";;
        W) export WKV="$OPTARG";;
        S) DEEPSTATE_STAGE="$OPTARG";;
        t) MAX_TRAINED_TOKENS="$OPTARG";;
        T) TERMINATE_LOSS="$OPTARG";;
        P) WANDB_PROJECT="$OPTARG";;
        W) WANDB="$OPTARG";;
        G) GPUS_PER_NODE="$OPTARG";;
        M) HAS_GROUP_NOMR="--has_group_norm";;
        \?) echo "无效的选项 -$OPTARG" >&2; exit 1;;
    esac
done

TRAIN_BATCH_SIZE=$((NNODES * GPUS_PER_NODE * MICRO_BSZ * ACCUMULATE_GRAD_BATCHES))
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
# --deepspeed_offload \

deepspeed \
    --num_nodes $NNODES \
    --num_gpus $GPUS_PER_NODE \
    train_scripts/train_hybrid_deepspeed.py \
    --deepspeed \
    $DEEPSPEED_OFFLOAD \
    $FULL_PARAMS \
    --deepspeed_stage $DEEPSTATE_STAGE \
    --config_file $CONFIG_FILE \
    --output_dir $OUTPUT_DIR \
    $PREPROCESSED_DATA \
    $RAW_DATA_DIR \
    $HAS_GROUP_NOMR \
    --num_devices $GPUS_PER_NODE \
    --num_nodes $NNODES \
    --micro_bsz $MICRO_BSZ \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs 1 \
    --wandb $WANDB \
    --run_name $WANDB_PROJECT \
    --grad_cp $GRAD_CP \
    --max_seq_length $MAX_LENGTH \
    --dropout 0.05 \
    --lr_init $LR_INIT \
    --lr_final $LR_FINAL \
    --warmup_steps $WARMUP_STEPS \
    --train_batch_size $TRAIN_BATCH_SIZE \
    --world_size $WORLD_SIZE \
    --save_per_batches 2000 \
    $CKPT_FILE \
    --stage $STAGE \
    --terminate_at_loss $TERMINATE_LOSS \
    --max_trained_tokens $MAX_TRAINED_TOKENS 