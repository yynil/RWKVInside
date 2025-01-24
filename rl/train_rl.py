import accelerate
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from rl_trainer import GRPOTrainer
from trl import GRPOConfig
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
dummy_data_file = "/home/yueyulin/data/MetaMathQA-395K-processed.jsonl"
# 创建数据集
dataset = datasets.load_dataset("json", data_files=dummy_data_file)["train"]

def reward_function(inputs):
    """计算奖励函数
    
    如果模型生成的答案中包含正确答案（格式为 \boxed{answer}），则给予奖励 1，否则为 0
    答案是否包含\nthinking\n \nthinking ends\n 对和
    \nanswer\n \nanswer ends\n对
    """
    rewards = []
    for input_data in inputs:
        completion = input_data['completion']
        ground_truth = input_data['ground_truth']
        reward = 0
        #如果包含\nthinking\n \nthinking ends\n
        index = completion.find("thinking\n")
        if index != -1:
            next_index = completion.find("thinking ends\n")
            if next_index != -1:
                reward += 0.2
            else:
                reward += 0.1
        #如果包含\nanswer\n \nanswer ends\n
        index = completion.find("answer\n")
        if index != -1:
            next_index = completion.find("answer ends\n")
            if next_index != -1:
                reward += 0.2
            else:
                reward += 0.1
        #如果正确答案在\boxed{}中，+0.2\boxed{3}
        boxed_ground_truth = f'\\boxed{{{ground_truth}}}'
        if boxed_ground_truth in completion:
            reward += 0.6
        rewards.append(reward)
    return torch.tensor(rewards, dtype=torch.float)

def main():
    accelerator = accelerate.Accelerator()
    # 初始化模型和tokenizer
    model_name = "/home/yueyulin/models/Qwen2.5-0.5B-Instruct/"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": accelerator.device},
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    # 配置训练参数
    training_args = GRPOConfig(
        output_dir="/tmp/math-solver-grpo",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_prompt_length=512,
        max_completion_length=1024,
        num_generations=4,  # 每个问题生成4个回答
        temperature=0.7,
        beta=0.1,  # KL散度的权重
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="tensorboard",
        no_cuda=False,
        fp16=False,
        bf16=True
    )
    print(dataset)
    print(dataset[0])
    # 初始化trainer
    trainer = GRPOTrainer(
        model=model,
        reward_function=reward_function,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer = accelerator.prepare(trainer)
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()

if __name__ == "__main__":
    main()