import os
import textwrap
from typing import Any, Callable, Optional, Union

from click import prompt
import torch
import torch.nn as nn
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollator,
    EvalPrediction,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, unwrap_model_for_generation
from grpo_config import GRPOConfig



if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel, nn.Module] = None,
        ref_model: Optional[PreTrainedModel] = None,
        reward_function: Optional[Callable[[Any], torch.Tensor]] = None,
        args: GRPOConfig = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_loss_func: Optional[Callable] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        preprocess_reward_inputs: Optional[Callable[[list, list, list], Any]] = None,
        chunk_size_to_calculate_kl: int = 1
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")
        self.chunk_size_to_calculate_kl = chunk_size_to_calculate_kl
        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    f"Invalid `torch_dtype`. Expected 'auto' or string representing torch.dtype, but got {torch_dtype}."
                )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        self.ref_model = ref_model

        # Processing class
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        # Reward function
        if reward_function is None:
            raise ValueError("reward_function must be provided")
        self.reward_function = reward_function

        # Custom reward input preprocessing
        self.preprocess_reward_inputs = preprocess_reward_inputs or self._default_preprocess_reward_inputs

        # Data loading and preprocessing
        if data_collator is None:
            def data_collator(features):
                return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
            use_cache=True
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True
        # model.gradient_checkpointing_enable()
        # Initialize metrics
        self._metrics = {"kl": [], "reward": [], "reward_std": []}

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        if self.ref_model is not None:
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False  # Ensure no gradients are stored
            self.ref_model.to(self.accelerator.device)
        
    def _default_preprocess_reward_inputs(self, prompts: list, completions: list, inputs: list) -> Any:
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
        processed_inputs = []
        for idx, input_data in enumerate(inputs):
            start_idx = idx * self.num_generations
            end_idx = (idx + 1) * self.num_generations
            
            # For each input, create G entries with the original input data
            # and the corresponding completion
            for i in range(start_idx, end_idx):
                entry = input_data.copy()  # Copy to avoid modifying original
                entry['completion'] = completions[i]
                processed_inputs.append(entry)
                
        return processed_inputs

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt","ground_truth","problem"]

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        # Process each prompt individually to ensure consistent batch sizes
        all_completions = []
        all_prompt_ids = []
        all_completion_ids = []
        
        for i in range(len(prompts_text)):
            # Process one prompt at a time
            prompt_inputs = self.processing_class(
                [prompts_text[i]], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            
            if self.max_prompt_length is not None:
                prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
                prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]

            # Generate completions with memory efficient settings
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                generation_config = self.generation_config
                generation_config.do_sample = True
                generation_config.use_cache = True
                with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                    print(f'device of unwrapped_model: {unwrapped_model.device}')
                    prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=generation_config)
                
                # Generate multiple completions for this single prompt
                # prompt_completion_ids = unwrapped_model.generate(
                #         **prompt_inputs,
                #         generation_config=generation_config
                #     )
                
            prompt_length = prompt_inputs["input_ids"].size(1)
            
            # Store prompt IDs for each generation
            prompt_ids_repeated = prompt_inputs["input_ids"].repeat(self.num_generations, 1)
            all_prompt_ids.append(prompt_ids_repeated)
            
            # Extract and store completion IDs
            batch_completion_ids = prompt_completion_ids[:, prompt_length:]
            all_completion_ids.append(batch_completion_ids)
            
            # Decode completions
            batch_completions = self.processing_class.batch_decode(batch_completion_ids, skip_special_tokens=True)
            all_completions.extend(batch_completions)
            
            # Clear memory after processing each prompt
            torch.cuda.empty_cache()
            # print(f'prompt_text: {prompts_text[i]}')
            # print(f'batch_completions: {batch_completions}')
        # Combine all results
        prompt_ids = torch.cat(all_prompt_ids, dim=0)
        completion_ids = torch.cat(all_completion_ids, dim=0)
        # Create full sequence for each prompt-completion pair
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        # Calculate log probabilities and KL divergence in chunks
        def get_chunked_per_token_logps(model, input_ids, chunk_size=self.chunk_size_to_calculate_kl):
            all_per_token_logps = []
            rank = self.accelerator.process_index
            for i in range(0, input_ids.size(0), chunk_size):
                chunk_input_ids = input_ids[i:i + chunk_size]
                # Free up memory before computing logits
                torch.cuda.empty_cache()
                
                logits = model(chunk_input_ids).logits
                logits = logits[:, :-1, :]
                chunk_ids = chunk_input_ids[:, 1:]
                
                chunk_per_token_logps = []
                for logits_row, input_ids_row in zip(logits, chunk_ids):
                    log_probs = logits_row.log_softmax(dim=-1)
                    token_log_prob = torch.gather(log_probs, dim=1, 
                                            index=input_ids_row.unsqueeze(1)).squeeze(1)
                    chunk_per_token_logps.append(token_log_prob)
                
                chunk_result = torch.stack(chunk_per_token_logps)
                all_per_token_logps.append(chunk_result)
                
                # Clear unnecessary tensors
                del logits, chunk_ids, chunk_per_token_logps
                torch.cuda.empty_cache()
                
            return torch.cat(all_per_token_logps, dim=0)

        # Calculate per-token log probabilities for model and reference model
        current_prompt_ids = prompt_ids[i:i + self.num_generations]  # 使用已经准备好的 all_prompt_ids
        prompt_completion_ids = torch.cat([current_prompt_ids, completion_ids], dim=1)
        # prompt_completion_ids = torch.cat([prompt_inputs["input_ids"], completion_ids], dim=1)
        per_token_logps = get_chunked_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            self.ref_model.eval()
            ref_per_token_logps = get_chunked_per_token_logps(self.ref_model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Calculate KL divergence in double-chunked manner
        batch_chunk_size = 4  # Number of sequences to process at once
        seq_chunk_size = 512  # Number of tokens to process at once within each sequence
        
        all_per_token_kl = []
        
        # Outer loop: Process sequences in chunks
        for i in range(0, per_token_logps.size(0), batch_chunk_size):
            batch_model_logps = per_token_logps[i:i + batch_chunk_size]  # [batch_chunk_size, seq_len]
            batch_ref_logps = ref_per_token_logps[i:i + batch_chunk_size]  # [batch_chunk_size, seq_len]
            
            # Initialize KL tensor for current batch of sequences
            seq_length = batch_model_logps.size(1)
            batch_kl = torch.zeros_like(batch_model_logps)
            
            # Inner loop: Process each sequence chunk by chunk
            for j in range(0, seq_length, seq_chunk_size):
                end_idx = min(j + seq_chunk_size, seq_length)
                
                # Get chunks of the current sequences
                chunk_model_logps = batch_model_logps[:, j:end_idx]  # [batch_chunk_size, seq_chunk_size]
                chunk_ref_logps = batch_ref_logps[:, j:end_idx]  # [batch_chunk_size, seq_chunk_size]
                
                # Calculate KL divergence for this chunk
                chunk_kl = torch.exp(chunk_ref_logps - chunk_model_logps) - \
                        (chunk_ref_logps - chunk_model_logps) - 1
                
                # Store result in the corresponding positions
                batch_kl[:, j:end_idx] = chunk_kl
                
                # Clear chunk tensors
                del chunk_model_logps, chunk_ref_logps, chunk_kl
                torch.cuda.empty_cache()
            
            all_per_token_kl.append(batch_kl)
            
            # Clear batch tensors
            del batch_model_logps, batch_ref_logps, batch_kl
            torch.cuda.empty_cache()
        
        per_token_kl = torch.cat(all_per_token_kl, dim=0)

        # Rest of the original compute_loss function remains the same
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Process rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        reward_inputs = self.preprocess_reward_inputs(prompts, all_completions, inputs)
        rewards = self.reward_function(reward_inputs)
        
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=device)
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        rewards = rewards.to(device)
        

        # Print generation samples if needed
        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero() and self.state.global_step > 0:
            print("\n=== Step {} Generation Samples ===".format(self.state.global_step))
            print(f"Input prompt:\n{prompts_text[0]}\n")
            ground_truth = inputs[0].get("ground_truth", "")
            print(f"Ground truth:\n{ground_truth}\n")
            start_idx = 0
            end_idx = self.num_generations
            print(f"Generated {self.num_generations} completions:")
            for i, (completion, reward) in enumerate(zip(all_completions[start_idx:end_idx], 
                                rewards[start_idx:end_idx])):
                print(f"\nCompletion {i+1}:")
                print(f"{completion}")
                print(f"Reward: {reward.item():.3f}")
                print("="*50)

        # Calculate final loss with normalized rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages.to(per_token_logps.device)
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Update metrics
        mean_reward_value = self.accelerator.gather_for_metrics(rewards).mean().item()
        self._metrics["reward"].append(mean_reward_value)
        mean_reward_std_dev = self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        self._metrics["reward_std"].append(mean_reward_std_dev)
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        mean_kl_metric = self.accelerator.gather_for_metrics(mean_kl).mean().item()
        self._metrics["kl"].append(mean_kl_metric)
        self.log_metrics({
            "loss": loss.item(),
            "mean_reward": mean_reward_value,
            "mean_reward_std_dev": mean_reward_std_dev,
            "kl": mean_kl_metric,
        })
        return loss
    
    def log_metrics(self, metrics):
        """Helper method to log metrics consistently"""
        # 更新内部指标存储
        for key, value in metrics.items():
            if key not in self._metrics:
                self._metrics[key] = []
            self._metrics[key].append(value)
        
        # 立即更新进度条显示的指标
        if self.state.global_step % self.args.logging_steps == 0:
            # 计算平均值
            avg_metrics = {
                key: sum(values[-self.args.logging_steps:])/len(values[-self.args.logging_steps:])
                for key, values in self._metrics.items()
            }
            # 使用 Trainer 的日志机制
            self.log(avg_metrics)
        
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # print(f'self._metrics: {self._metrics}')
        metrics = {key: sum(val)/len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics = {key: [] for key in self._metrics}