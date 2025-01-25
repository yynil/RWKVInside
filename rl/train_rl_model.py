import accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.state import AcceleratorState
from transformers import HfArgumentParser
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from rl_trainer import GRPOTrainer
from grpo_config import GRPOConfig
import logging
from dataclasses import dataclass, field
from typing import Optional
import os

@dataclass
class ScriptArguments:
    """
    Custom arguments for the training script
    """
    # Data and model arguments
    data_file: str = field(
        default=None, metadata={"help": "Path to the training data file (JSONL format)"}
    )
    model_name: str = field(
        default=None, metadata={"help": "Path or name of the pretrained model"}
    )
    output_dir: str = field(
        default=None, metadata={"help": "Directory to save the trained model"}
    )
    
    # Training hyperparameters
    num_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Training batch size per device"}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of gradient accumulation steps"}
    )
    learning_rate: float = field(
        default=1e-5, metadata={"help": "Learning rate"}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight decay for AdamW optimizer"}
    )
    
    # Generation parameters
    max_prompt_length: int = field(
        default=512, metadata={"help": "Maximum length for input prompts"}
    )
    max_completion_length: int = field(
        default=1024, metadata={"help": "Maximum length for generated completions"}
    )
    num_generations: int = field(
        default=4, metadata={"help": "Number of generations per prompt"}
    )
    temperature: float = field(
        default=0.7, metadata={"help": "Temperature for generation sampling"}
    )
    beta: float = field(
        default=0.1, metadata={"help": "KL divergence weight"}
    )
    
    # Other training settings
    seed: int = field(
        default=42, metadata={"help": "Random seed for reproducibility"}
    )
    logging_steps: int = field(
        default=100, metadata={"help": "Number of steps between logging"}
    )
    chunk_size_to_calculate_kl: int = field(
        default=1, metadata={"help": "Chunk size to calculate KL divergence"}
    )
    wandb_project: str = field(
        default="grpo-training", metadata={"help": "Name of the W&B project"}
    )
    wandb_run_name: str = field(
        default=None, metadata={"help": "Name of the W&B run"}
    )
    
def setup_logging():
    """Configure logging with a consistent format"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def reward_function(inputs):
    """Calculate rewards based on model outputs"""
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
        index = completion.find("thinking\n")
        if index != -1:
            next_index = completion.find("thinking ends\n")
            if next_index != -1:
                reward += 0.2
            else:
                reward += 0.1
                
        # Check for answer structure
        index = completion.find("answer\n")
        if index != -1:
            next_index = completion.find("answer ends\n")
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
                #convert the boxed ground truth to a number
                try:
                    value_boxed_ground_truth = float(boxed_ground_truth)
                    if abs(value_ground_truth - value_boxed_ground_truth) < 1e-6:
                        reward += 0.6
                except ValueError:
                    pass
        else:    
            boxed_ground_truth = f'\\boxed{{{ground_truth}}}'
            if boxed_ground_truth in completion:
                reward += 0.6
            
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float)

def main():
    # Parse arguments
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    # Set up project configuration
    project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Set up logging
    setup_logging()
    
    # Log basic information
    accelerator.print(f"Using accelerate version {accelerate.__version__}")
    accelerator.print(f"Distributed environment: {AcceleratorState().distributed_type}")
    
    # Load dataset
    dataset = datasets.load_dataset("json", data_files=args.data_file)["train"]
    
    
    
    # Initialize model and tokenizer with proper dtype
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Configure GRPO training arguments
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        beta=args.beta,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="wandb",
        run_name=args.wandb_run_name,
        bf16=True
    )
    os.environ["WANDB_PROJECT"] = args.wandb_project
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        ref_model=reference_model,
        reward_function=reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        chunk_size_to_calculate_kl=args.chunk_size_to_calculate_kl
    )
    
    # Prepare for distributed training
    trainer = accelerator.prepare(trainer)
    
    # Train the model
    trainer.train()
    
    # Save the final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        trainer.save_model()
        accelerator.print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()