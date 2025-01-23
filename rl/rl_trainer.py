import os
import textwrap
from typing import Any, Callable, Optional, Union

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
from trl import GRPOConfig

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

class GRPOTrainer(Trainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel, nn.Module] = None,
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
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

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
        if peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

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
        )
        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True
        
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
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

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
        # print(f'inputs: {len(inputs)}')
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        # print(prompts_text)
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        if self.max_prompt_length is not None:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
            # print(f'prompt_inputs input_ids: {prompt_inputs["input_ids"].shape}')
            # print(f'prompt_inputs attention_mask: {prompt_inputs["attention_mask"].shape}')

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)
        prompt_length = prompt_inputs["input_ids"].size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        # Get per-token log probabilities
        def get_per_token_logps(model, input_ids):
            logits = model(input_ids).logits
            logits = logits[:, :-1, :]
            input_ids = input_ids[:, 1:]
            per_token_logps = []
            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)

        per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = get_per_token_logps(self.ref_model, prompt_completion_ids)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = get_per_token_logps(model, prompt_completion_ids)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Mask after first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(f'completions: {completions}')

        # Prepare inputs for reward computation
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        reward_inputs = self.preprocess_reward_inputs(prompts, completions, inputs)
        
        # Compute rewards using the provided reward function
        rewards = self.reward_function(reward_inputs)
        
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=device)
        if rewards.dim() == 2:
            rewards = rewards.squeeze(-1)
        rewards = rewards.to(self.accelerator.device)
        
        # 在生成完成后打印日志
        if self.state.global_step % self.args.logging_steps == 0 and self.is_world_process_zero():  # Only print on rank 0
            print("\n=== Step {} Generation Samples ===".format(self.state.global_step))
            # 取batch中的第一个样本来打印
            print(f"Input prompt:\n{prompts_text[0]}\n")
            ground_truth = inputs[0].get("ground_truth", "")
            print(f"Ground truth:\n{ground_truth}\n")
            # 打印该样本的所有生成结果和对应rewards
            start_idx = 0
            end_idx = self.num_generations
            print(f"Generated {self.num_generations} completions:")
            for i, (completion, reward) in enumerate(zip(completions[start_idx:end_idx], 
                                rewards[start_idx:end_idx])):
                print(f"\nCompletion {i+1}:")
                print(f"{completion}")
                print(f"Reward: {reward.item():.3f}")
                print("="*50)

        # Compute grouped rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize rewards
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantages = advantages.to(per_token_logps.device)
        advantages = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(advantages - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Log metrics
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        # print(f'self._metrics: {self._metrics}')
        metrics = {key: sum(val)/len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics = {key: [] for key in self._metrics}