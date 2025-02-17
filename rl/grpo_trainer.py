import os
from pyexpat import model
import tokenize
from tracemalloc import stop
from numpy import pad
from sympy import per
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
from deepspeed.runtime.zero.stage3 import unwrap_model_for_generation
log_level = os.getenv("LOG_LEVEL", "INFO").upper()  # Default to INFO if not set
logging.basicConfig(level=getattr(logging, log_level, logging.INFO),
                    format="%(asctime)s [PID:%(process)d] [TID:%(thread)d] %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S.%f")

logger = logging.getLogger(__name__)


@torch.compile(fullgraph=True)
def grpo_loss_with_old_logps(
    logps: torch.Tensor, 
    ref_logps: torch.Tensor,
    old_logps: torch.Tensor, 
    pad_mask: torch.Tensor,
    logits_to_keep: int, 
    rewards: torch.Tensor,
    beta: float = 0.2,
    epsilon: float = 0.2
):
    """
    Compute the GRPO (Group Relative Policy Optimization) loss.
    Args:
        logps: Log probabilities of the current policy
        ref_logps: Log probabilities of the reference policy
        old_logps: Log probabilities of the old policy
        pad_mask: Mask for padding tokens
        logits_to_keep: Number of logits to keep
        rewards: Rewards for each token
        beta: KL divergence weight
        epsilon: Clipping parameter for importance weights
    Returns:
    The GRPO loss
    """
    B = logps.shape[0]
    assert B > 1, "Batch * Num generations should be greater than 1"
    
    rewards_shaped = rewards.view(-1, B)#B,num_generations
    advantages = (rewards_shaped - rewards_shaped.mean(dim=1, keepdim=True)) / \
                (rewards_shaped.std(dim=1, keepdim=True) + 1e-8)
    advantages = advantages.view(-1)#B*num_generations
    # Calculate the per - token KL divergence
    per_token_kl = torch.exp(ref_logps - logps) - (ref_logps - logps) - 1

    # Calculate the ratio of probabilities (importance weights)
    # Importance weights are calculated as exp(log_pi_theta - log_pi_theta_old)
    importance_weights = torch.exp(logps - old_logps)

    # Clip the importance weights to the range [1 - epsilon, 1 + epsilon]
    importance_weights_clipped = torch.clamp(importance_weights, 1 - epsilon, 1 + epsilon)

    # Create a completion mask. It checks which positions are valid based on logits_to_keep
    completion_mask = torch.arange(logits_to_keep, device=logps.device)[None, :] >= 0

    # Combine the completion mask and padding mask
    completion_mask = completion_mask & pad_mask  # Ensure matching shape

    # Add an extra dimension to advantages to match the shape for element - wise multiplication
    advantages = advantages.unsqueeze(1)

    # Calculate the per - token loss. It takes the minimum of the unclipped and clipped importance weights
    # and subtracts the KL divergence term weighted by beta, then multiplies by the completion mask
    token_loss = -(torch.min(advantages * importance_weights, advantages * importance_weights_clipped) - beta * per_token_kl) * completion_mask

    # Calculate the final loss by summing the token losses and normalizing by the number of valid tokens
    loss = -token_loss.sum() / completion_mask.sum()

    return loss

#borrow from TRL utility
@torch.compile(fullgraph=True)
def selective_log_softmax_old(logits, index):
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
def selective_log_softmax(logits, index, chunk_size=1):
    """Memory-efficient implementation of selective log softmax.
    
    Args:
        logits: Input logits tensor of shape (batch_size, seq_len, vocab_size)
        index: Index tensor of shape (batch_size, seq_len) 
        chunk_size: Size of chunks to process at once to reduce memory usage
        
    Returns:
        Log probabilities for selected indices
    """
    device = logits.device
    batch_size, seq_len, vocab_size = logits.shape
    all_per_token_logps = []
    
    # Process in batch chunks
    for i in range(0, batch_size, chunk_size):
        chunk_end = min(i + chunk_size, batch_size)
        chunk_logits = logits[i:chunk_end]  # [chunk_size, seq_len, vocab_size]
        chunk_index = index[i:chunk_end]
        
        with torch.amp.autocast('cuda'):
            # 计算全局 max，保持数值稳定性
            max_logits = chunk_logits.max()  # 全局最大值
            chunk_logits = chunk_logits - max_logits
            
            # 分片计算 logsumexp
            log_denominator = torch.zeros(chunk_logits.shape[:-1], device=device)
            for j in range(0, vocab_size, 1024):  # 按词表维度分片
                j_end = min(j + 1024, vocab_size)
                log_denominator += torch.exp(chunk_logits[..., j:j_end]).sum(dim=-1)
            
            log_denominator = torch.log(log_denominator) + max_logits
            
            # 获取选定索引的 logits
            selected_logits = chunk_logits.gather(
                dim=-1, 
                index=chunk_index.unsqueeze(-1)
            ).squeeze(-1) + max_logits
            
            # 计算最终的 log probabilities
            chunk_log_probs = selected_logits - log_denominator
        
        all_per_token_logps.append(chunk_log_probs)
        
        # 清理内存
        del chunk_logits, log_denominator, selected_logits
        torch.cuda.empty_cache()
            
    return torch.cat(all_per_token_logps, dim=0)

@time_function
def log_samples(prompt, ground_truth, completion, reward, step, num_generations, local_rank):
    """Log training samples and rewards"""
    reward = reward.tolist()
    if local_rank <= 0:
        logger.info(f"Step {step}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Ground truth: {ground_truth}")
        for i in range(num_generations):
            logger.info(f"Completion {i}: {completion[i]}")
            logger.info(f"Reward {i}: {reward[i]}")
            logger.info("----------------")
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
    updates_mu: int = 4


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
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            output_logits=True,
            return_dict_in_generate=True
        )
        print(f'tokenizer {self.tokenizer},stop_strings {self.tokenizer.eos_token},pad_token_id {self.tokenizer.pad_token_id}')
        self.preprocess_reward_inputs = preprocess_reward_inputs
        self.count = 0
    @time_function
    def _generate_completions(self, prompt_inputs):
        """Generate completions efficiently with DeepSpeed Engine."""
        input_ids = prompt_inputs["input_ids"]
        attention_mask = prompt_inputs.get("attention_mask")
        if self.args.ds_stage == 3:
            with unwrap_model_for_generation(self.model_engine) as unwrapped_model:
                unwrapped_model.eval()
                with torch.no_grad():
                    generations = unwrapped_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=self.generation_config
                    )
                unwrapped_model.train()
        else:
            self.model_engine.module.eval()
            with torch.no_grad():
                generations = self.model_engine.module.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config
                )
            self.model_engine.module.train()
        return generations


    @time_function
    def _batch_chunked_forward(self, model, input_ids, chunk_batch=2):
        """Compute model logits in a memory-efficient way by chunking along the last dimension."""
        torch.cuda.empty_cache()
        all_logits = []
        for i in range(0, input_ids.size(0), chunk_batch):
            batch_input_ids = input_ids[i:i+chunk_batch]
            attention_mask = batch_input_ids != self.tokenizer.pad_token_id
            logits = model(batch_input_ids,attention_mask=attention_mask,use_cache=False).logits
            logits = logits[:, :-1, :]#chunk_batch,T-1,V exclude the last logit
            all_logits.append(logits)
            torch.cuda.empty_cache()
        return torch.cat(all_logits, dim=0)#B,T-1,V
    
    
    @time_function
    def train_step(self, batch):
        """Execute single training step."""
        
        try:
            if self.args.local_rank == 0:
                logger.debug(f"Step {self.count}")
            # Process input batch
            prompts = batch["prompt"]
            prompts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
            # Tokenize
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.args.max_prompt_length,
                add_special_tokens=False,
                padding_side="left"
            ).to(self.model_engine.device)#B,T

            # Generate completions
            if self.args.local_rank == 0:
                logger.debug(f"start to generate completions with prompt_inputs shape {prompt_inputs['input_ids'].shape}")
            generations = self._generate_completions(prompt_inputs)
            if self.args.local_rank == 0:
                logger.debug("finish generating completions")
            # generation_logits = generations.logits #B,T,V This is the old policy logits
            #cat logits to B,T,V from tuple of B,V
            generations, completion_ids, logits_to_keep, old_logps, completions = self._prepare_old_logps(prompt_inputs, generations)
            # Calculate rewards
            rewards = self.reward_function(
                self.preprocess_reward_inputs(prompts, completions, batch)
            )
            rewards = rewards.to(self.model_engine.device)

            # Normalize rewards
            rewards_shaped = rewards.view(-1, self.args.num_generations)#B,num_generations
            advantages = (rewards_shaped - rewards_shaped.mean(dim=1, keepdim=True)) / \
                        (rewards_shaped.std(dim=1, keepdim=True) + 1e-8)
            advantages = advantages.view(-1)#B*num_generations
            if self.args.local_rank == 0:
                logger.debug(f"Start to compute KL divergence")
                
            ref_logps = self._prepare_ref_logps(generations, logits_to_keep)
            
            self.count += 1    
            #calculate model_logits
            for i in range(self.args.updates_mu):
                self.model_engine.train()
                if self.args.local_rank == 0:
                    logger.debug(f"GRPO interation {i}")
                #calculate model_logits
                model_logps, per_token_kl, completion_mask = self._prepare_inputs(generations, completion_ids, logits_to_keep, ref_logps)
                loss = self._compute_loss(completion_ids, logits_to_keep, old_logps, rewards, ref_logps, model_logps)

                # Compute loss
                # epsilon = 0.2  # You can adjust this hyperparameter

                # # Calculate the ratio of probabilities (importance weights)
                # # Assuming you have access to the log probabilities of the current policy (log_pi_theta)
                # # and the log probabilities of the old policy (log_pi_theta_old)
                # # Here I am assuming that token_kl is the kl divergence D(pi_theta || pi_ref), and log_pi_theta_old - log_pi_theta = D(pi_theta || pi_ref)
                # importance_weights =  torch.exp(model_logps-old_logps) #exp(log_pi_theta - log_pi_theta_old) # This needs to be computed

                # # Clip the importance weights
                # importance_weights_clipped = torch.clamp(importance_weights, 1 - epsilon, 1 + epsilon)
                # # Calculate the GRPO loss using the minimum of the clipped and unclipped importance weights
                # completion_mask = torch.arange(logits_to_keep, device=generations.device)[None, :] >= 0
                # pad_mask = completion_ids != self.tokenizer.pad_token_id  # Shift to match log_probs dimension
                # completion_mask = completion_mask & pad_mask  # Ensure matching shape
                # advantages = advantages.unsqueeze(1)
                # # advantages = torch.exp(model_logps - model_logps.detach()) * advantages
                # token_loss = -(torch.min(advantages * importance_weights, advantages * importance_weights_clipped) - self.args.beta * per_token_kl) * completion_mask
                
                
                # # token_loss = -(advantages.unsqueeze(1) - self.args.beta * token_kl) * completion_mask
                # loss = -token_loss.sum() / completion_mask.sum()
                if self.count % self.args.logging_steps == 0:
                    log_samples(prompts[0], batch["ground_truth"][0], completions, rewards, self.count, self.args.num_generations, self.args.local_rank)
                average_generation_length = completion_mask.sum(dim=1).float().mean()
                if self.args.local_rank == 0:
                    logger.debug("Finish training step")
                mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
                #tmp codes to save inputs and the corresponding loss value
                # if self.count <= 3 and i == 0 and self.args.local_rank == 0:
                #     #save inputs and loss in pk file
                #     import pickle
                #     with open(f'inputs_{self.count}.pk','wb') as f:
                #         obj = {
                #             'ref_logps':ref_logps,
                #             'old_logps':old_logps,
                #             'epsilon':epsilon,
                #             'completion_mask':completion_mask,
                #             'advantages':advantages,
                #             'loss':loss,
                #             'rewards':rewards,
                #         }
                #         pickle.dump(obj,f)
                #         print(f'save inputs and loss in inputs_{self.count}.pk')
                self.model_engine.backward(loss)
                self.model_engine.step()
            return loss,rewards.mean(),rewards.std(),mean_kl,average_generation_length
            
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
    @time_function
    def _prepare_old_logps(self, prompt_inputs, generations):
        with torch.no_grad():
            generation_logits = torch.cat([logit.unsqueeze(1) for logit in generations.logits], dim=1)
            generations = generations.sequences#B*self.args.num_generations,T+GENERATION_LENGTH
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = generations[:, prompt_length:]#B*self.args.num_generations,GENERATION_LENGTH
            logits_to_keep = completion_ids.size(1) #GENERATION_LENGTH
            generation_logits = generation_logits[:, -logits_to_keep:]#B,GENERATION_LENGTH,V
            old_logps = selective_log_softmax(generation_logits[:,-logits_to_keep:,:], generations[:,-logits_to_keep:])
            if self.args.local_rank == 0:
                logger.debug(f'old_logits shape {generation_logits.shape},old_logps shape {old_logps.shape},require grad {old_logps.requires_grad},old_logits require grad {generation_logits.requires_grad}')
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
            if self.args.local_rank == 0:
                logger.debug(f"Completions: {completions}")
        return generations,completion_ids,logits_to_keep,old_logps,completions

    def _prepare_ref_logps(self, generations, logits_to_keep):
        with torch.no_grad():
            ref_logits = self._batch_chunked_forward(self.ref_model_engine, generations, chunk_batch=self.args.batch_chunk_size) 
            ref_logits = ref_logits[:,-logits_to_keep:,:]#B,logits_to_keep,V
            ref_logps = selective_log_softmax(ref_logits, generations[:,-logits_to_keep:])
            if self.args.local_rank == 0:
                logger.debug(f"ref_logits shape {ref_logits.shape},ref_logps shape {ref_logps.shape},require grad {ref_logps.requires_grad},ref_logits require grad {ref_logits.requires_grad}")
            del ref_logits
        return ref_logps
    @time_function
    def _compute_loss(self, completion_ids, logits_to_keep, old_logps, rewards, ref_logps, model_logps):
        return grpo_loss_with_old_logps(
                    model_logps,
                    ref_logps,
                    old_logps,
                    completion_ids != self.tokenizer.pad_token_id,
                    logits_to_keep,
                    rewards,
                    self.args.beta,
                    epsilon=0.2
                )
    @time_function
    def _prepare_inputs(self, generations, completion_ids, logits_to_keep, ref_logps):
        model_logits = self._batch_chunked_forward(self.model_engine, generations, chunk_batch=self.args.batch_chunk_size)
        model_logits = model_logits[:,-logits_to_keep:,:]#B,logits_to_keep,V        
        model_logps = selective_log_softmax(model_logits, generations[:,-logits_to_keep:])   
        per_token_kl = torch.exp(ref_logps - model_logps) - (ref_logps - model_logps) - 1
        if self.args.local_rank == 0:
            logger.debug(f"Start to compute rewards and loss")
        completion_mask = torch.arange(logits_to_keep, device=generations.device)[None, :] >= 0
        pad_mask = completion_ids != self.tokenizer.pad_token_id
        completion_mask = completion_mask & pad_mask
        return model_logps,per_token_kl,completion_mask
