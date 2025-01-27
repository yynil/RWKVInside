import os
from pyexpat import model
import torch
import torch.nn as nn
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
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True
        )
        self.preprocess_reward_inputs = preprocess_reward_inputs
        self.count = 0
    @time_function
    def _generate_completions(self, prompt_inputs):
        """Generate completions efficiently with DeepSpeed Engine."""
        self.model_engine.eval()
        with torch.no_grad():
            generations = self.model_engine.module.generate(
                input_ids=prompt_inputs["input_ids"],
                attention_mask=prompt_inputs.get("attention_mask"),
                generation_config=self.generation_config,
                synced_gpus=True
            )
        self.model_engine.train()
        return generations
    @time_function
    def _compute_per_token_logps(self, model_engine, input_ids):
        """Compute per-token log probabilities for a model."""
        with torch.no_grad():
            logits = model_engine(input_ids).logits[:, :-1, :]
            
        shift_labels = input_ids[:, 1:]
        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        return token_log_probs
    @time_function
    def _compute_kl_divergence(self, generations, prompt_length):
        """Compute KL divergence between model and reference model distributions."""
        model_log_probs = self._compute_per_token_logps(self.model_engine, generations)
        ref_log_probs = self._compute_per_token_logps(self.ref_model_engine, generations)
        
        token_kl = torch.exp(ref_log_probs - model_log_probs) - \
                  (ref_log_probs - model_log_probs) - 1
                  
        # Create completion mask
        completion_mask = torch.arange(
            model_log_probs.size(1),
            device=generations.device
        )[None, :] >= (prompt_length - 1)
        completion_mask = completion_mask.expand(generations.size(0), -1)
        mean_kl = ((token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        return token_kl, completion_mask,mean_kl
    @time_function
    def train_step(self, batch):
        """Execute single training step."""
        self.model_engine.train()
        
        try:
            # Process input batch
            prompts = batch["prompt"]
            prompts = [self.tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
            
            # Tokenize
            prompt_inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.args.max_prompt_length
            ).to(self.model_engine.device)

            # Generate completions
            generations = self._generate_completions(prompt_inputs)
            prompt_length = prompt_inputs["input_ids"].size(1)
            completion_ids = generations[:, prompt_length:]
            completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            # Compute KL divergence
            token_kl, completion_mask,mean_kl = self._compute_kl_divergence(generations, prompt_length)

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

            # Compute loss
            token_loss = -(advantages.unsqueeze(1) - self.args.beta * token_kl) * completion_mask
            loss = token_loss.sum() / completion_mask.sum()
            self.count += 1
            if self.count % self.args.logging_steps == 0:
                log_samples(prompts[0], batch["ground_truth"][0], completions, rewards, self.count, self.args.num_generations, self.args.local_rank)
            return loss,rewards.mean(),rewards.std(),mean_kl
            
        except Exception as e:
            logging.error(f"Error in train_step:")
            logging.error(traceback.format_exc())
            raise
