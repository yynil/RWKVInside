import os
from pyexpat import model
import tokenize
from tracemalloc import stop
from numpy import pad
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributed as dist
import deepspeed
from typing import Optional, Union, Any, Callable
from dataclasses import dataclass
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig
)
import json
import logging
import traceback
from profiler import time_function
log_level = os.getenv("LOG_LEVEL", "INFO").upper()  # Default to INFO if not set
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s [PID:%(process)d] [TID:%(thread)d] %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S.%f")

logger = logging.getLogger(__name__)

#borrow from TRL utility
def selective_log_softmax(logits, index):
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.logsumexp(logits, dim=-1)
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps
@time_function
def log_samples(prompt, ground_truth, completion, reward, step, num_generations, local_rank):
    """Log training samples and rewards"""
    reward = reward.tolist()
    if local_rank <= 0:
        logging.info(f"Step {step}")
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Ground truth: {ground_truth}")
        for i in range(num_generations):
            logging.info(f"Completion {i}: {completion[i]}")
            logging.info(f"Reward {i}: {reward[i]}")
            logging.info("----------------")
class ConversationDataCollator:
    """Custom collator for conversation data."""
    def __init__(self):
        pass
    
    def __call__(self, batch):
        processed_batch = {
            "problem": [],
            "ground_truth": [],
            "prompt": []
        }
        
        for item in batch:
            if isinstance(item, str):
                item = json.loads(item)
            
            if isinstance(item["problem"], list):
                processed_batch["problem"].extend(item["problem"])
                processed_batch["ground_truth"].extend(item["ground_truth"])
            else:
                processed_batch["problem"].append(item["problem"])
                processed_batch["ground_truth"].append(item["ground_truth"])

            if isinstance(item["prompt"], list):
                if isinstance(item["prompt"][0], list):
                    processed_batch["prompt"].extend(item["prompt"])
                else:
                    processed_batch["prompt"].append(item["prompt"])
            else:
                processed_batch["prompt"].append(item["prompt"])
        
        return processed_batch
            
@dataclass
class GRPOConfig:
    """Configuration for GRPO training"""
    output_dir: str 
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    num_generations: int = 4
    temperature: float = 0.7
    beta: float = 0.1
    logging_steps: int = 100
    save_steps: int = 500
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    local_rank: int = -1
    chunk_size: int = 1024
    batch_chunk_size: int = 2
    ds_stage : int = 3


class GRPOTrainer:
    def __init__(
        self,
        model_engine,
        ref_model_engine,
        args,
        tokenizer,
        reward_function,
        preprocess_reward_inputs=None
    ):
        
        self.model_engine = model_engine
        self.ref_model_engine = ref_model_engine
        self.args = args
        self.tokenizer = tokenizer
        self.reward_function = reward_function  
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.args.max_completion_length,
            do_sample=True,
            temperature=self.args.temperature,
            num_return_sequences=self.args.num_generations,
            use_cache=True,
            tokenizer=self.tokenizer,
            stop_strings=[self.tokenizer.eos_token],
            pad_token_id=self.tokenizer.pad_token_id
        )
        print(f'tokenizer {self.tokenizer},stop_strings {self.tokenizer.eos_token},pad_token_id {self.tokenizer.pad_token_id}')
        self.preprocess_reward_inputs = preprocess_reward_inputs
        self.count = 0
    @time_function
    def _generate_completions(self, prompt_inputs):
        """Generate completions efficiently with DeepSpeed Engine."""
        if self.args.ds_stage == 3:
            with deepspeed.zero.unwrap_model_for_generation(self.model_engine) as unwrapped_model:
                unwrapped_model.eval()
                with torch.no_grad():
                    generations = unwrapped_model.generate(
                        input_ids=prompt_inputs["input_ids"],
                        attention_mask=prompt_inputs.get("attention_mask"),
                        generation_config=self.generation_config,
                        synced_gpus=True,
                        tokenizer=self.tokenizer
                    )
                unwrapped_model.train()
        else:
            self.model_engine.module.eval()
            with torch.no_grad():
                generations = self.model_engine.module.generate(
                    input_ids=prompt_inputs["input_ids"],
                    attention_mask=prompt_inputs.get("attention_mask"),
                    generation_config=self.generation_config,
                    synced_gpus=True,
                    tokenizer=self.tokenizer
                )
            self.model_engine.module.train()
        return generations
    @time_function
    def _chunked_log_softmax(self,logits, chunk_size=1024):
        """
        Compute log_softmax in a memory-efficient way by chunking along the last dimension.
        
        Args:
            logits (Tensor): The input logits of shape (B, T, Vocab_size).
            chunk_size (int): The size of chunks to split the last dimension.
        
        Returns:
            Tensor: Log-softmax computed over the last dimension.
        """
        logger.debug(f"Logits shape: {logits.shape}")
        max_logits = torch.max(logits, dim=-1, keepdim=True)[0]  # 防止溢出的最大值
        log_probs = torch.empty_like(logits)  # 预分配 log_probs 避免内存重复分配
        
        sum_exp_logits = torch.zeros_like(max_logits)
        for i in range(0, logits.shape[-1], chunk_size):
            chunk = logits[..., i:i+chunk_size] - max_logits
            exp_chunk = torch.exp(chunk)
            sum_exp_logits += torch.sum(exp_chunk, dim=-1, keepdim=True)
        
        log_sum_exp = torch.log(sum_exp_logits)
        for i in range(0, logits.shape[-1], chunk_size):
            log_probs[..., i:i+chunk_size] = logits[..., i:i+chunk_size] - max_logits - log_sum_exp
        
        return log_probs

    @time_function
    def _compute_per_token_logps_chunked(self, model_engine, input_ids, chunk_size=1024, chunk_batch=2):
        """Compute per-token log probabilities with reduced memory usage."""
        """Compute per-token log probabilities with reduced memory usage."""
        with torch.no_grad():
            all_log_probs = []
            for i in range(0, input_ids.size(0), chunk_batch):
                batch_input_ids = input_ids[i:i+chunk_batch]
                logits = model_engine(batch_input_ids).logits[:, :-1, :]
                log_probs = self._chunked_log_softmax(logits, chunk_size=chunk_size)
                all_log_probs.append(log_probs)
        return torch.cat(all_log_probs, dim=0)
    
    def _compute_per_token_kl_chunked(self,model_engine,ref_engine,input_ids,chunk_batch=2,chunk_size=1024):
        """Compute per-token KL divergence between model and reference model distributions."""
        with torch.no_grad():
            all_token_kl = []
            for i in range(0,input_ids.size(0),chunk_batch):
                batch_input_ids = input_ids[i:i+chunk_batch]
                model_logits = model_engine(batch_input_ids).logits[:, :-1, :]
                torch.cuda.empty_cache()
                ref_logits = ref_engine(batch_input_ids).logits[:, :-1, :]  
                torch.cuda.empty_cache()
                model_log_probs = self._chunked_log_softmax(model_logits, chunk_size=chunk_size)
                ref_log_probs = self._chunked_log_softmax(ref_logits, chunk_size=chunk_size)
                torch.cuda.empty_cache() 
                B,T,V = model_log_probs.shape
                chunked_token_kl = torch.empty(B,T,device=model_log_probs.device,dtype=model_log_probs.dtype)
                for j in range(0,T,chunk_size):
                    model_chunk = model_log_probs[:,j:j+chunk_size,:]
                    ref_chunk = ref_log_probs[:,j:j+chunk_size,:]
                    token_kl = (torch.exp(ref_chunk - model_chunk) - (ref_chunk - model_chunk) - 1).sum(dim=-1)#B,chunk_size
                    chunked_token_kl[:,j:j+chunk_size] = token_kl
                # token_kl = torch.cat(chunked_token_kl,dim=1).sum(dim=-1)
                # token_kl = (torch.exp(ref_log_probs - model_log_probs) - (ref_log_probs - model_log_probs) - 1).sum(dim=-1)
                torch.cuda.empty_cache()
                # all_token_kl.append(token_kl)
                all_token_kl.append(chunked_token_kl)
        return torch.cat(all_token_kl, dim=0)
    @time_function
    def _compute_kl_divergence(self, generations, prompt_length, chunk_batch=2):
        """Compute KL divergence between model and reference model distributions."""
        # model_log_probs = self._compute_per_token_logps_chunked(self.model_engine, generations)
        # ref_log_probs = self._compute_per_token_logps_chunked(self.ref_model_engine, generations)
        
        # token_kl = (torch.exp(ref_log_probs - model_log_probs) - \
        #           (ref_log_probs - model_log_probs) - 1).sum(dim=-1)
        logger.debug(f"INPUTIDS shape: {generations.shape}")
        token_kl = self._compute_per_token_kl_chunked(self.model_engine,self.ref_model_engine,generations,chunk_size=self.args.chunk_size,chunk_batch=chunk_batch)
        logger.debug(f"Token KL shape: {token_kl.shape}")
       # Create completion mask
        batch_size,seq_length = generations.shape[:2]
        completion_mask = torch.arange(seq_length, device=generations.device)[None, :] >= (prompt_length - 1)
        logger.debug(f"Completion mask shape: {completion_mask.shape}")
        # Mask out pad tokens
        pad_mask = generations[:, 1:] != self.tokenizer.pad_token_id  # Shift to match log_probs dimension
        logger.debug(f"Pad mask shape: {pad_mask.shape}")
        completion_mask = completion_mask[:, 1:] & pad_mask  # Ensure matching shape
        logger.debug(f"Completion mask shape: {completion_mask.shape}")
        # completion_mask = completion_mask.expand(generations.size(0), -1)
        logger.debug(f"Token KL shape: {token_kl.shape}")
        mean_kl = ((token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return token_kl, completion_mask,mean_kl
    
    @time_function
    def _batch_chunked_forward(self, model, input_ids, chunk_batch=2):
        """Compute model logits in a memory-efficient way by chunking along the last dimension."""
        torch.cuda.empty_cache()
        all_logits = []
        for i in range(0, input_ids.size(0), chunk_batch):
            batch_input_ids = input_ids[i:i+chunk_batch]
            logger.debug(f"Batch {i} forward {batch_input_ids.shape}")
            logits = model(batch_input_ids).logits
            logits = logits[:, :-1, :]#chunk_batch,T-1,V exclude the last logit
            all_logits.append(logits)
            torch.cuda.empty_cache()
        return torch.cat(all_logits, dim=0)#B,T-1,V
    
    @time_function
    def train_step(self, batch):
        """Execute single training step."""
        self.model_engine.train()
        
        try:
            if self.args.local_rank == 0:
                logging.info(f"Step {self.count}")
            # Process input batch
            prompts = batch["prompt"]
            logger.debug(f"Prompts: {prompts}")
            prompts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
            logger.debug(f"Processed prompts: {prompts}")
            # Tokenize
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_prompt_length
            ).to(self.model_engine.device)

            # Generate completions
            if self.args.local_rank == 0:
                logging.info("start to generate completions")
            generations = self._generate_completions(prompt_inputs)
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = generations[:, prompt_length:]#B,T
            logits_to_keep = completion_ids.size(1) #T-promt_length
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            logger.debug(f"Completions: {completions}")
            # Calculate rewards
            rewards = self.reward_function(
                self.preprocess_reward_inputs(prompts, completions, batch)
            )
            rewards = rewards.to(self.model_engine.device)

            # Normalize rewards
            rewards_shaped = rewards.view(-1, self.args.num_generations)
            advantages = (rewards_shaped - rewards_shaped.mean(dim=1, keepdim=True)) / \
                        (rewards_shaped.std(dim=1, keepdim=True) + 1e-8)
            advantages = advantages.view(-1)
            if self.args.local_rank == 0:
                logging.info(f"Start to compute KL divergence")
                
            #rewrite the GRPO loss
            #calculate ref_logits
            logger.debug(f'ref_model generations shape {generations.shape}')
            with torch.no_grad():
                ref_logits = self._batch_chunked_forward(self.ref_model_engine, generations, chunk_batch=self.args.batch_chunk_size) 
            ref_logits = ref_logits[:,-logits_to_keep:,:]#B,logits_to_keep,V
            ref_logps = selective_log_softmax(ref_logits, generations[:,-logits_to_keep:])
            #calculate model_logits
            logger.debug(f'model generations shape {generations.shape}')
            model_logits = self._batch_chunked_forward(self.model_engine, generations, chunk_batch=self.args.batch_chunk_size)
            model_logits = model_logits[:,-logits_to_keep:,:]#B,logits_to_keep,V        
            model_logps = selective_log_softmax(model_logits, generations[:,-logits_to_keep:])    
            log_probs_diff = model_logps - ref_logps
            exp_log_probs_diff = torch.exp(log_probs_diff)
            # per_token_kl = (1/exp_log_probs_diff) + log_probs_diff - 1
            per_token_kl =  exp_log_probs_diff + (-log_probs_diff) - 1 #torch.exp(- log_probs_diff) + log_probs_diff - 1
            if self.args.local_rank == 0:
                logging.info(f"Start to compute rewards and loss")
            

            # Compute loss
            epsilon = 0.2  # You can adjust this hyperparameter

            # Calculate the ratio of probabilities (importance weights)
            # Assuming you have access to the log probabilities of the current policy (log_pi_theta)
            # and the log probabilities of the old policy (log_pi_theta_old)
            # Here I am assuming that token_kl is the kl divergence D(pi_theta || pi_ref), and log_pi_theta_old - log_pi_theta = D(pi_theta || pi_ref)
            importance_weights =  exp_log_probs_diff #exp(log_pi_theta - log_pi_theta_old) # This needs to be computed

            # Clip the importance weights
            importance_weights_clipped = torch.clamp(importance_weights, 1 - epsilon, 1 + epsilon)
            # Calculate the GRPO loss using the minimum of the clipped and unclipped importance weights
            batch_size,seq_length = generations.shape[:2]
            completion_mask = torch.arange(seq_length, device=generations.device)[None, :] >= (prompt_length - 1)
            pad_mask = generations[:, 1:] != self.tokenizer.pad_token_id  # Shift to match log_probs dimension
            completion_mask = completion_mask[:, 1:] & pad_mask  # Ensure matching shape
            advantages = advantages.unsqueeze(1)
            token_loss = -(torch.min(advantages * importance_weights, advantages * importance_weights_clipped) - self.args.beta * per_token_kl) * completion_mask
            
            # token_loss = -(advantages.unsqueeze(1) - self.args.beta * token_kl) * completion_mask
            loss = -token_loss.sum() / completion_mask.sum()
            self.count += 1
            if self.count % self.args.logging_steps == 0:
                log_samples(prompts[0], batch["ground_truth"][0], completions, rewards, self.count, self.args.num_generations, self.args.local_rank)
            average_generation_length = completion_mask.sum(dim=1).float().mean()
            if self.args.local_rank == 0:
                logging.info("Finish training step")
            return loss,rewards.mean(),rewards.std(),0,average_generation_length
            
            # per_token_kl = torch.exp(ref_logps - model_logps) - (ref_logps - model_logps) - 1
            
            # Compute KL divergence
            # token_kl, completion_mask,mean_kl = self._compute_kl_divergence(generations, prompt_length, chunk_batch=self.args.batch_chunk_size)
            # if self.args.local_rank == 0:
            #     logging.info(f"Start to compute rewards and loss")
            

            # # Compute loss
            # epsilon = 0.2  # You can adjust this hyperparameter

            # # Calculate the ratio of probabilities (importance weights)
            # # Assuming you have access to the log probabilities of the current policy (log_pi_theta)
            # # and the log probabilities of the old policy (log_pi_theta_old)
            # # Here I am assuming that token_kl is the kl divergence D(pi_theta || pi_ref), and log_pi_theta_old - log_pi_theta = D(pi_theta || pi_ref)
            # importance_weights =  - token_kl #exp(log_pi_theta - log_pi_theta_old) # This needs to be computed

            # # Clip the importance weights
            # importance_weights_clipped = torch.clamp(importance_weights, 1 - epsilon, 1 + epsilon)
            #  # Calculate the GRPO loss using the minimum of the clipped and unclipped importance weights
            # token_loss = -(torch.min(advantages.unsqueeze(1) * importance_weights, advantages.unsqueeze(1) * importance_weights_clipped) - self.args.beta * token_kl) * completion_mask
            
            # # token_loss = -(advantages.unsqueeze(1) - self.args.beta * token_kl) * completion_mask
            # loss = token_loss.sum() / completion_mask.sum()
            # self.count += 1
            # if self.count % self.args.logging_steps == 0:
            #     log_samples(prompts[0], batch["ground_truth"][0], completions, rewards, self.count, self.args.num_generations, self.args.local_rank)
            # average_generation_length = completion_mask.sum(dim=1).float().mean()
            # if self.args.local_rank == 0:
            #     logging.info("Finish training step")
            # return loss,rewards.mean(),rewards.std(),mean_kl,average_generation_length
            
        except Exception as e:
            logging.error(f"Error in train_step:")
            logging.error(traceback.format_exc())
            raise
