from calendar import c
from multiprocessing import process
from multiprocessing.util import is_abstract_socket_namespace
import os
from regex import T
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import deepspeed
import datasets
import wandb
from tqdm import tqdm
from transformers import HfArgumentParser, AutoTokenizer,AutoModelForCausalLM
from dataclasses import dataclass, field
import logging
import json
from typing import Optional
from grpo_trainer import GRPOTrainer, GRPOConfig, ConversationDataCollator
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from functools import partial
from profiler import timer
from utilities import compare_latex_numbers
import time  # 添加这一行

def preprocess_reward_inputs(prompts: list, completions: list, inputs: list):
    """Default preprocessing for reward inputs.
    
    Args:
        prompts: List of prompts
        completions: List of model generated completions
        inputs: Original input data containing additional information (e.g., problem, ground_truth)
        
    Returns:
        Preprocessed inputs for reward computation, with each input repeated num_generations times
        along with corresponding completions.
    """
    # Repeat each input G times to match with generated completions
    num_generations = len(completions) // len(prompts)
    processed_inputs = []
    for i in range(len(prompts)):
        start_idx = i * num_generations
        end_idx = (i + 1) * num_generations
        ground_truth = inputs['ground_truth'][i]
        for j in range(start_idx, end_idx):
            completion = completions[j]
            processed_inputs.append({
                "prompt": prompts[i],
                "completion": completion,
                "ground_truth": ground_truth
            })
    return processed_inputs
    
def reward_function(inputs):
    """Calculate rewards based on model outputs"""
    # logging.info(f'reward function inputs: {inputs}')
    rewards = []
    for input_data in inputs:
        completion = input_data['completion']
        ground_truth = input_data['ground_truth']
        #check if the ground_truth is a number
        is_number_ground_truth = False
        try:
            ground_truth = ground_truth.strip().replace(" ", "").replace(",", "")
            value_ground_truth = float(ground_truth)
            is_number_ground_truth = True
        except ValueError:
            is_number_ground_truth = False
        reward = 0
        
        # Check for thinking structure
        index = completion.find("<think>")
        if index != -1:
            next_index = completion.find("</think>")
            if next_index != -1:
                reward += 0.2
            else:
                reward += 0.1
                
        # Check for correct answer in \boxed{} format
        if is_number_ground_truth:
            #found the \boxed{} format
            index_of_boxed = completion.find("\\boxed{")
            if index_of_boxed != -1:
                next_index_of_boxed = completion.find("}", index_of_boxed)
                boxed_ground_truth = completion[index_of_boxed+len("\\boxed{"):next_index_of_boxed]
                if compare_latex_numbers(boxed_ground_truth,ground_truth):
                    reward += 0.6
        else:    
            boxed_ground_truth = f'\\boxed{{{ground_truth}}}'
            if boxed_ground_truth in completion:
                reward += 0.6
            
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float,requires_grad=True)
@dataclass
class ScriptArguments:
    """Command line arguments for training script"""
    data_file: str = field(
        default=None,
        metadata={"help": "Path to training data file (JSONL format)"}
    )
    model_name: str = field(
        default=None,
        metadata={"help": "Path or name of pretrained model"}
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "Directory to save trained model"}
    )
    deepspeed_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to DeepSpeed config file"}
    )
    num_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={"help": "Training batch size per device"}
    )
    learning_rate: float = field(
        default=1e-5,
        metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay"}
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps"}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of gradient accumulation steps"}
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum length for input prompts"}
    )
    max_completion_length: int = field(
        default=1024,
        metadata={"help": "Maximum length for generated completions"}
    )
    num_generations: int = field(
        default=4,
        metadata={"help": "Number of generations per prompt"}
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Temperature for generation sampling"}
    )
    beta: float = field(
        default=0.1,
        metadata={"help": "KL divergence weight"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Number of steps between logging"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Number of steps between saving checkpoints"}
    )
    local_rank: int = field(
        default=-1,
        metadata={"help": "Local rank for distributed training"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed"}
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "Name of W&B project"}
    )
    wandb_run_name: str = field(
        default=None,
        metadata={"help": "Name of W&B run"}
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Use gradient checkpointing"}
    )
    
    chunk_size : int = field(
        default=1024,
        metadata={"help": "chunk size"}
    )
    
    batch_chunk_size: int = field(
        default=2,
        metadata={"help": "batch chunk size"}
    )
    
    ds_stage: int = field(
        default=3,
        metadata={"help": "DeepSpeed stage"}
    )

    ds_param_offload : bool = field(
        default=True,
        metadata={"help": "DeepSpeed parameter offload"}
    )
    
    ds_optimizer_offload : bool = field(
        default=True,
        metadata={"help": "DeepSpeed optimizer offload"}
    )
    
    is_att_tuning_only : bool = field(
        default=False,
        metadata={"help": "Attention tuning only"}
    )
def setup_logging(local_rank):
    """Configure logging"""
    if local_rank <= 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

def load_dataset(data_file):
    """Load and preprocess the training dataset"""
    dataset = datasets.load_dataset("json", data_files=data_file)["train"]
    return dataset

def configure_optimizer(model, args):
    lr_1x = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lr_1x.add(n)

    lr_1x = sorted(list(lr_1x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

    if args.ds_optimizer_offload:
        from deepspeed.ops.adam import DeepSpeedCPUAdam
        optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.999),  bias_correction=True, adamw_mode=True, amsgrad=False)
    else:
        from deepspeed.ops.adam import FusedAdam
        optimizer = FusedAdam(optim_groups, lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, bias_correction=True, adam_w_mode=True, amsgrad=False)
  
    return optimizer

def save_checkpoint(model_engine, output_dir, epoch, step,logger):
    """Save model checkpoint"""
    if os.path.exists(output_dir):
        if model_engine.local_rank == 0:
            checkpoints = os.listdir(output_dir)
            #only list the directories   
            checkpoints = [f for f in checkpoints if os.path.isdir(os.path.join(output_dir, f))]
            #sort by creation time  
            checkpoints.sort(key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
            if len(checkpoints) > 2:
                print(f'deleting older checkpoints {checkpoints[0]}')
                import shutil
                shutil.rmtree(os.path.join(output_dir, checkpoints[0]))    
    output_dir = f"{output_dir}/epoch_{epoch}_step_{step}"
    print(f'saving checkpoint to {output_dir}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    model_engine.save_checkpoint(output_dir)

def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Setup environment variables
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    world_size = int(os.getenv('WORLD_SIZE', '1'))
    is_main_process = local_rank == 0
    device = torch.device(f'cuda:{local_rank}')
    
    # Setup logging
    setup_logging(local_rank)
    logger = logging.getLogger(__name__)
    
    if is_main_process:
        logger.info("Starting GRPO training with DeepSpeed")
        logger.info(f"Arguments: {args}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize tokenizer
    if is_main_process:
        logger.info(f"Loading tokenizer from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    
    # Load dataset
    if is_main_process:
        logger.info(f"Loading dataset from {args.data_file}")
    dataset = load_dataset(args.data_file)
    
    # Setup data loading
    if is_main_process:
        logger.info(f"Creating DataLoader with batch size {args.per_device_train_batch_size}, world size {world_size}")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=args.seed
    )
    
    data_collator = ConversationDataCollator()
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator
    )
    
    # Load DeepSpeed config
    if args.deepspeed_config:
        if is_main_process:
            logger.info(f"Loading DeepSpeed config from {args.deepspeed_config}")
        with open(args.deepspeed_config, 'r') as f:
            ds_config = json.load(f)
    else:
        # Default DeepSpeed config is using ZeRO-3 with CPU offload
        if is_main_process:
            logger.info("Using default DeepSpeed config")
        train_batch_size = args.per_device_train_batch_size * world_size* args.gradient_accumulation_steps
        ds_config = {
                "distributed_backend": "nccl",
                "train_batch_size": train_batch_size,
                "bf16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": args.ds_stage,
                    "stage3_max_live_parameters": 1e9,
                    "stage3_max_reuse_distance": 1e9,
                    "stage3_prefetch_bucket_size": 5e6,
                    "memory_efficient_linear": True,
                    "stage3_param_persistence_threshold": 1e4,
                    "offload_param": {
                        "device": "cpu",
                        "pin_memory": True,
                        "buffer_count": 4,
                        "buffer_size": 1e8
                    },
                    "offload_optimizer": {
                        "device": "cpu",
                        "pin_memory": True,
                        "buffer_count": 4
                    },
                    "allgather_partitions": True,
                    "reduce_scatter": True,
                    "reduce_bucket_size": 5e6,
                    "overlap_comm": True,
                    "contiguous_gradients": True
                },
                "zero_force_ds_cpu_initialization": True,
                "gradient_checkpointing": args.gradient_checkpointing,
                "dump_state": True
            }
        
    #Init model with deepspeed
    if is_main_process:
        logger.info(f"Initializing model with DeepSpeed config")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    if is_main_process:
        logger.info(f'Enable gradient checkpointing: {args.gradient_checkpointing}')
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.train()
    for n,p in model.named_parameters():
        if args.is_att_tuning_only:
            if 'self_attn' not in n:
                p.requires_grad = False
        else:
            p.requires_grad = True
    if is_main_process:
        logger.info(f'start configuring optimizer')
    optimizer = configure_optimizer(model, args)
    # Initialize DeepSpeed for main model
    model_ds_config = ds_config.copy()
    if not args.ds_param_offload:
        del model_ds_config["zero_optimization"]["offload_param"]
    if not args.ds_optimizer_offload:
        del model_ds_config["zero_optimization"]["offload_optimizer"]
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            config=model_ds_config,
            model_parameters=model.parameters(),
            optimizer=optimizer
    )
    timer.initialize_with_engine(model_engine)
    if is_main_process:
        logger.info("Model initialized")
    del model
    # Initialize reference model
    if is_main_process:
        logger.info(f"Initializing reference model")
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    ref_model.eval()
    #freeze all parameters of reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_ds_config = ds_config.copy()
    if args.ds_stage == 3:
        del ref_ds_config["zero_optimization"]["offload_optimizer"] 
    else:
        # 对参考模型禁用 ZeRO 优化
        ref_ds_config["zero_optimization"]["stage"] = 0
    ref_model_engine, _, _, _ = deepspeed.initialize(
            model=ref_model,
            config=ref_ds_config
    )
    del ref_model
    if is_main_process:
        logger.info("Reference model initialized")
    if is_main_process:
        logger.info(f'current gpu memory AFTER setting model and ref model: {torch.cuda.memory_summary(device=None, abbreviated=False)}')
        logger.info(f'Start training with {len(dataloader)} batches')
    
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        local_rank=args.local_rank,
        chunk_size=args.chunk_size,
        batch_chunk_size=args.batch_chunk_size,
        ds_stage=args.ds_stage
    )
    trainer = GRPOTrainer(
        model_engine,
        ref_model_engine,
        training_args,
        tokenizer,
        reward_function,
        preprocess_reward_inputs=preprocess_reward_inputs
    )
    total_loss = 0
    total_reward_mean = 0 
    total_reward_std = 0
    total_steps = 0
    if is_main_process:
        from tqdm import tqdm
        pbar = tqdm(total=len(dataloader))
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    #delete the output_dir if it exists
    if os.path.exists(args.output_dir):
        os.rmdir(args.output_dir)
        
    for epoch in range(args.num_epochs):
        if is_main_process:
            logger.info(f"Epoch {epoch} starts training")
        # 使用时间戳生成随机种子
        time_seed = int(time.time() * 1000) & 0xffffffff  # 获取毫秒级时间戳并转换为32位整数
        sampler.set_epoch(time_seed)  # 使用时间戳作为种子
        
        for batch_idx,batch in enumerate(dataloader):
            loss,reward_mean,reward_std,mean_kl,average_generation_length = trainer.train_step(batch)
            model_engine.backward(loss)
            model_engine.step()
            if batch_idx % args.save_steps == 0 and batch_idx > 0:
                if (args.ds_stage != 3 and is_main_process) or (args.ds_stage == 3):
                    save_checkpoint(model_engine, args.output_dir, epoch, batch_idx,logger)
            # 累计统计
            if is_main_process:
                total_loss += loss.item()
                total_reward_mean += reward_mean.item()
                total_reward_std += reward_std.item()
                total_steps += 1
                
                # 计算平均值
                avg_loss = total_loss / total_steps
                avg_reward_mean = total_reward_mean / total_steps
                avg_reward_std = total_reward_std / total_steps
                
                # 记录到wandb
                wandb.log({
                    "loss": loss.item(),
                    "reward_mean": reward_mean.item(),
                    "reward_std": reward_std.item(),
                    "kl": mean_kl.item(),
                    "avg_generation_length": average_generation_length.item(),
                    "avg_loss": avg_loss,
                    "avg_reward_mean": avg_reward_mean,
                    "avg_reward_std": avg_reward_std,
                    "epoch": epoch,
                    "step": total_steps
                })
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': loss.item(),
                    'avg_loss': avg_loss,
                    'reward_mean': reward_mean.item(),
                    'avg_reward': avg_reward_mean,
                    'kl': mean_kl.item()
                })
        #save checkpoint at the end of each epoch
        if (args.ds_stage != 3 and is_main_process) or (args.ds_stage == 3):
            epoch_checkpoint_dir = f"{args.output_dir}/epoch_{epoch}"
            if not os.path.exists(epoch_checkpoint_dir):
                os.makedirs(epoch_checkpoint_dir)
            print(f'saving checkpoint to {epoch_checkpoint_dir}')
            model_engine.save_checkpoint(epoch_checkpoint_dir)
    # 训练结束后关闭wandb
    if is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()