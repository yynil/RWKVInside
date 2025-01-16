import json
from math import trunc
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModelForCausalLM
from tqdm import tqdm
import os
import pandas as pd
import random
teacher_device = "cuda:0"
student_device = "cuda:1"
def load_test_samples(jsonl_path, max_samples):
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    print(f"Loaded {len(samples)} samples from {jsonl_path} and randomly selecting {max_samples} samples.")
    samples = random.sample(samples, min(len(samples), max_samples))
    return samples

def get_model_distribution(model, tokenizer, text, device='cuda'):
    inputs = tokenizer([text], 
                       max_length=2048,
                       truncation=True,
                       return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return F.softmax(outputs.logits, dim=-1).float().cpu()

def compute_metrics(teacher_probs, student_probs):
    # 计算每个位置的KL散度
    position_kl = F.kl_div(
        student_probs.log(), 
        teacher_probs,
        reduction='none'
    ).sum(-1)  # 对词表维度求和
    
    # 计算平均KL散度（按位置和batch维度平均）
    avg_kl = position_kl.mean().item()
    
    # 计算最后一个位置的KL散度
    last_position_kl = position_kl[0, -1].item()
    
    metrics = {
        'kl_div': avg_kl,  # 所有位置的平均KL散度
        'last_position_kl': last_position_kl,  # 最后一个位置的KL散度
        'sequence_length': teacher_probs.size(1),  # 序列长度
        'max_position_kl': position_kl.max().item(),  # 最大位置KL散度
        'min_position_kl': position_kl.min().item(),  # 最小位置KL散度
    }
    
    # Top-k overlap (仍然只看最后一个位置)
    for k in [1, 5, 10, 50]:
        t_topk = torch.topk(teacher_probs[0, -1], k)[1]
        s_topk = torch.topk(student_probs[0, -1], k)[1]
        overlap = len(set(t_topk.tolist()) & set(s_topk.tolist()))
        metrics[f'top{k}_overlap'] = overlap / k
        
    return metrics

def plot_full_distribution(teacher_probs, student_probs, save_path, top_k=None):
    """绘制完整概率分布对比图"""
    plt.figure(figsize=(15, 10))
    
    # 获取最后一个时间步的概率分布
    t_probs = teacher_probs[0, -1].cpu()
    s_probs = student_probs[0, -1].cpu()
    
    # 按概率值排序
    t_sorted, _ = torch.sort(t_probs, descending=True)
    s_sorted, _ = torch.sort(s_probs, descending=True)
    
    if top_k is None:
        top_k = len(t_sorted)
    
    x = range(top_k)
    
    # 主图：线性尺度
    plt.subplot(2, 1, 1)
    plt.plot(x, t_sorted[:top_k], 'b-', label='Teacher', alpha=0.7)
    plt.plot(x, s_sorted[:top_k], 'r-', label='Student', alpha=0.7)
    plt.title('Probability Distribution Comparison (Linear Scale)')
    plt.xlabel('Token Index (Sorted by Probability)')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    # 子图：对数尺度
    plt.subplot(2, 1, 2)
    plt.semilogy(x, t_sorted[:top_k], 'b-', label='Teacher', alpha=0.7)
    plt.semilogy(x, s_sorted[:top_k], 'r-', label='Student', alpha=0.7)
    plt.title('Probability Distribution Comparison (Log Scale)')
    plt.xlabel('Token Index (Sorted by Probability)')
    plt.ylabel('Probability (log scale)')
    
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def convert_conversation_to_text(conversation):
    """convert conversations to chatml
    {"deita_score":10.7623962717,"rw_score":4.3125,"id":"generate_from_topic_and_persona","instruction":"","lang":"zh","length":17,"conversations":[{"from":"human","value":"您能介绍一下航空航天在环境保护方面的具体应用吗？"},{"from":"gpt","value":"航空航天在环境保护方面有许多具体应用。首先，航空航天技术可以用于监测和预测气候变化。卫星观测和大气探测仪器可以收集大量的气象数据，帮助科学家更好地理解气候变化的模式和趋势。其次，航空航天技术可以用于监测和管理地球上的自然资源。卫星遥感和空中激光雷达可以提供关于森林覆盖、水资源、土地利用等方面的详细信息，从而帮助决策者更好地保护和管理这些资源。此外，航空航天技术还可以用于环境灾害的预警和应急响应。卫星图像和空中监测系统可以实时监测火灾、洪水、地震等灾害，提供及时的信息支持和救援指导。总之，航空航天在环境保护方面的具体应用可以为我们提供更多的数据和工具，帮助我们更好地了解和保护我们的地球。"}]}
    <|im_start|>user\n您能介绍一下航空航天在环境保护方面的具体应用吗？<|im_end|><|im_start|>assistant\n航空航天在环境保护方面有许多具体应用。首先，航空航天技术可以用于监测和预测气候变化。卫星观测和大气探测仪器可以收集大量的气象数据，帮助科学家更好地理解气候变化的模式和趋势。其次，航空航天技术可以用于监测和管理地球上的自然资源。卫星遥感和空中激光雷达可以提供关于森林覆盖、水资源、土地利用等方面的详细信息，从而帮助决策者更好地保护和管理这些资源。此外，航空航天技术还可以用于环境灾害的预警和应急响应。卫星图像和空中监测系统可以实时监测火灾、洪水、地震等灾害，提供及时的信息支持和救援指导。总之，航空航天在环境保护方面的具体应用可以为我们提供更多的数据和工具，帮助我们更好地了解和保护我们的地球。\n<|im_end|>

    Args:
        conversation (_type_): _description_
    """
    text = ""
    for conv in conversation:
        if conv['from'] == 'human':
            text += f"<|im_start|>user\n{conv['value']}<|im_end|>"
        elif conv['from'] == 'gpt':
            text += f"<|im_start|>assistant\n{conv['value']}<|im_end|>"
    return text

def evaluate_models(teacher_model, student_model, tokenizer, test_samples):
    results = []
    
    for sample in tqdm(test_samples):
        if 'text' in sample:
            text = sample['text'].strip()
            if len(text) == 0:
                print(f"Skipping empty text sample.:{sample}")
                continue
        elif 'conversations' in sample:
            conversations = sample['conversations']
            """
            {"deita_score":10.7623962717,"rw_score":4.3125,"id":"generate_from_topic_and_persona","instruction":"","lang":"zh","length":17,"conversations":[{"from":"human","value":"您能介绍一下航空航天在环境保护方面的具体应用吗？"},{"from":"gpt","value":"航空航天在环境保护方面有许多具体应用。首先，航空航天技术可以用于监测和预测气候变化。卫星观测和大气探测仪器可以收集大量的气象数据，帮助科学家更好地理解气候变化的模式和趋势。其次，航空航天技术可以用于监测和管理地球上的自然资源。卫星遥感和空中激光雷达可以提供关于森林覆盖、水资源、土地利用等方面的详细信息，从而帮助决策者更好地保护和管理这些资源。此外，航空航天技术还可以用于环境灾害的预警和应急响应。卫星图像和空中监测系统可以实时监测火灾、洪水、地震等灾害，提供及时的信息支持和救援指导。总之，航空航天在环境保护方面的具体应用可以为我们提供更多的数据和工具，帮助我们更好地了解和保护我们的地球。"}]}
            """
            text = convert_conversation_to_text(conversations)
        teacher_probs = get_model_distribution(teacher_model, tokenizer, text,device=teacher_device)
        student_probs = get_model_distribution(student_model, tokenizer, text,device=student_device)
        
        # 保存分布图
        plot_full_distribution(
            teacher_probs, 
            student_probs, 
            f'plots/dist_{len(results)}.png'
        )
        
        # 计算全词表的指标
        metrics = compute_metrics(teacher_probs, student_probs)
        
        # 添加更多top-k评估
        for k in [100, 500, 1000, 5000]:
            t_topk = torch.topk(teacher_probs[0, -1], k)[1]
            s_topk = torch.topk(student_probs[0, -1], k)[1]
            overlap = len(set(t_topk.tolist()) & set(s_topk.tolist()))
            metrics[f'top{k}_overlap'] = overlap / k
        
        metrics['text'] = text
        results.append(metrics)
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, required=True)
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, default='test/test_samples.jsonl')
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    with torch.no_grad():
        teacher_model = AutoModelForCausalLM.from_pretrained(args.teacher_model).half()
        student_model = AutoModelForCausalLM.from_pretrained(args.student_model).half()
        teacher_model.to(teacher_device)
        student_model.to(student_device)
        
    # 加载模型和数据
    test_samples = load_test_samples(args.input_jsonl, args.max_samples)
    
    # 评估并可视化
    results_df = evaluate_models(teacher_model, student_model, tokenizer, test_samples)
    
    # 打印汇总结果
    print("\n评估结果汇总:")
    print(results_df[results_df.columns[:-1]].describe())