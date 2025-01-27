import deepspeed
import torch
import torch.nn.functional as F
import logging
import cupy as cp
from cupy.cuda import nccl
import json
import torch
from torch.optim import Adam
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from profiler import time_function

def rank0_print(*args, **kwargs):
    if deepspeed.comm.get_rank() == 0:
        print(*args, **kwargs)
        

@time_function
def dpo_train_step(model, ref_model, batch, args):
    """
    Implements DPO training step following TRL's implementation
    """
    # Helper for concatenating tensors
    @time_function
    def concatenate_inputs(batch, pad_token_id):
        """
        Concatenate inputs handling variable lengths, ensuring each prompt+completion pair
        is concatenated before padding
        """
        batch_size = batch["prompt_input_ids"].shape[0]
        
        # Initialize lists for chosen and rejected sequences separately
        chosen_sequences = []
        chosen_masks = []
        rejected_sequences = []
        rejected_masks = []
        prompt_lens = []  # Store prompt lengths for later use
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Get prompt length for this sample (excluding padding)
            prompt_len = int(torch.sum(batch["prompt_attention_mask"][i]).item())  # Ensure integer
            prompt_lens.append(prompt_len)
            prompt_ids = batch["prompt_input_ids"][i, :prompt_len]
            
            # Process chosen completion
            chosen_len = int(torch.sum(batch["chosen_attention_mask"][i]).item())  # Ensure integer
            chosen_ids = batch["chosen_input_ids"][i, :chosen_len]
            
            # Process rejected completion
            rejected_len = int(torch.sum(batch["rejected_attention_mask"][i]).item())  # Ensure integer
            rejected_ids = batch["rejected_input_ids"][i, :rejected_len]
            
            # Concatenate prompt with chosen and rejected
            chosen_seq = torch.cat([prompt_ids, chosen_ids])
            rejected_seq = torch.cat([prompt_ids, rejected_ids])
            
            # Create attention masks
            chosen_mask = torch.ones(len(chosen_seq), device=chosen_seq.device)
            rejected_mask = torch.ones(len(rejected_seq), device=rejected_seq.device)
            
            # Store sequences in separate lists
            chosen_sequences.append(chosen_seq)
            chosen_masks.append(chosen_mask)
            rejected_sequences.append(rejected_seq)
            rejected_masks.append(rejected_mask)
        
        # Combine in correct order: all chosen, then all rejected
        sequences = chosen_sequences + rejected_sequences
        attention_masks = chosen_masks + rejected_masks
        
        # Find max length across all sequences
        max_len = max(len(seq) for seq in sequences)
        
        # Create padded tensors
        padded_input_ids = torch.full(
            (batch_size * 2, max_len),
            pad_token_id,
            dtype=sequences[0].dtype,
            device=sequences[0].device
        )
        attention_mask = torch.zeros(
            (batch_size * 2, max_len),
            dtype=attention_masks[0].dtype,
            device=attention_masks[0].device
        )
        
        # Fill in sequences and masks
        sequence_lens = []  # Store sequence lengths for later use
        for i, (seq, mask) in enumerate(zip(sequences, attention_masks)):
            seq_len = len(seq)
            sequence_lens.append(seq_len)
            padded_input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = mask
        
        # Create loss mask (0 for prompt, 1 for completion)
        loss_mask = torch.zeros_like(attention_mask)
        for i in range(batch_size * 2):
            # For chosen sequences (0 to batch_size-1)
            # For rejected sequences (batch_size to 2*batch_size-1)
            base_idx = i if i < batch_size else i - batch_size
            prompt_len = prompt_lens[base_idx]  # Use stored integer
            seq_len = sequence_lens[i]  # Use stored integer
            loss_mask[i, prompt_len:seq_len] = 1
        
        # Shift loss mask for next-token prediction
        loss_mask = loss_mask[:, 1:].bool()
        
        # Truncate if needed
        if args.max_seq_length > 0:
            padded_input_ids = padded_input_ids[:, :args.max_seq_length]
            attention_mask = attention_mask[:, :args.max_seq_length]
            loss_mask = loss_mask[:, :args.max_seq_length-1]
        #For RWKV-7 Chunk, len of input_ids must be 16x, so we need to pad the input_ids to 16x
        length = padded_input_ids.shape[1]
        #we pad the input_ids to args.max_seq_length 
        #if args.max_seq_length % 16 == 0, it will satisfy RWKV-7 chunk requirement
        if length < args.max_seq_length:
            padded_input_ids = F.pad(padded_input_ids, (0, args.max_seq_length-length), value=pad_token_id)
            attention_mask = F.pad(attention_mask, (0, args.max_seq_length-length), value=0)
            loss_mask = F.pad(loss_mask, (0, args.max_seq_length-length), value=0)
        
        '''
        if args.local_rank == 0:
            print("\nSequence order verification:")
            print(f"Total sequences: {len(sequences)}")
            print(f"Prompt lengths: {prompt_lens}")
            print(f"Sequence lengths: {sequence_lens}")
            print(f"Final padded length: {max_len}")
            
            # Print actual sequences for verification
            print("\nFirst chosen and rejected for first sample:")
            print(f"Chosen (sample 1): {padded_input_ids[0, :sequence_lens[0]].tolist()}")
            print(f"Rejected (sample 1): {padded_input_ids[batch_size, :sequence_lens[batch_size]].tolist()}")
            print(f"Whole input_ids (sample 1) Chosen: {padded_input_ids[0].tolist()}")
            print(f"Whole attention_mask (sample 1) Chosen: {attention_mask[0].tolist()}")
            print(f"Whole loss_mask (sample 1) Chosen: {loss_mask[0].tolist()}")
            print(f"Whole input_ids (sample 1) Rejected: {padded_input_ids[batch_size].tolist()}")
            print(f"Whole attention_mask (sample 1) Rejected: {attention_mask[batch_size].tolist()}")
            print(f"Whole loss_mask (sample 1) Rejected: {loss_mask[batch_size].tolist()}")
        '''
        return padded_input_ids, attention_mask, loss_mask

    # Get model outputs for concatenated sequences
    @time_function
    def get_model_outputs(model, input_ids, attention_mask, loss_mask):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get logits and prepare labels
        logits = outputs.logits[:, :-1]  # Remove last logit
        labels = input_ids[:, 1:].clone()  # Shift right for next-token prediction
        
        # Calculate per-token log probabilities
        per_token_logps = torch.gather(
            logits.log_softmax(-1),
            dim=2,
            index=labels.unsqueeze(2)
        ).squeeze(2)
        
        # Apply loss mask and sum
        per_token_logps = per_token_logps * loss_mask
        return per_token_logps.sum(dim=-1), logits

    # Concatenate inputs
    input_ids, attention_mask, loss_mask = concatenate_inputs(batch,args.pad_id)
    batch_size = batch["prompt_input_ids"].shape[0]

    # Get policy model outputs
    policy_logps, policy_logits = get_model_outputs(model, input_ids, attention_mask, loss_mask)
    chosen_policy_logps = policy_logps[:batch_size]
    rejected_policy_logps = policy_logps[batch_size:]

    # Get reference model outputs
    with torch.no_grad():
        ref_logps, ref_logits = get_model_outputs(ref_model, input_ids, attention_mask, loss_mask)
        chosen_ref_logps = ref_logps[:batch_size]
        rejected_ref_logps = ref_logps[batch_size:]

    # Calculate logits/rewards
    chosen_logratios = chosen_policy_logps - chosen_ref_logps
    rejected_logratios = rejected_policy_logps - rejected_ref_logps
    logits = chosen_logratios - rejected_logratios
    
    # Calculate loss based on type
    if args.loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(args.dpo_beta * logits) * (1 - args.label_smoothing)
            - F.logsigmoid(-args.dpo_beta * logits) * args.label_smoothing
        )
    elif args.loss_type == "hinge":
        losses = torch.relu(1 - args.dpo_beta * logits)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")

    # Calculate metrics
    chosen_rewards = (chosen_policy_logps - chosen_ref_logps).detach()
    rejected_rewards = (rejected_policy_logps - rejected_ref_logps).detach()
    reward_accuracies = (chosen_rewards > rejected_rewards).float()

    metrics = {
        "rewards/chosen": chosen_rewards.mean(),
        "rewards/rejected": rejected_rewards.mean(),
        "rewards/accuracies": reward_accuracies.mean(),
        "rewards/margins": (chosen_rewards - rejected_rewards).mean(),
        "logits/chosen": policy_logits[:batch_size][loss_mask[:batch_size]].mean(),
        "logits/rejected": policy_logits[batch_size:][loss_mask[batch_size:]].mean(),
        "logps/chosen": chosen_policy_logps.mean(),
        "logps/rejected": rejected_policy_logps.mean()
    }

    return losses.mean(), metrics

@time_function
def train_step(model, batch, args, teacher_engine=None, tokenizer=None):
    # print(batch)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask'].to(torch.int32)
    if 'labels' in batch:
        labels = batch['labels']
        # 验证labels的维度
        if labels.shape != input_ids.shape:
            raise ValueError(f"Labels shape {labels.shape} doesn't match input_ids shape {input_ids.shape}")
    else:
        # 直接创建左移的labels
        labels = torch.cat([input_ids[:, 1:], 
                          torch.full((input_ids.shape[0], 1), 
                                   tokenizer.pad_token_id, 
                                   device=input_ids.device)], dim=1)
        

    # attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).to(input_ids.device)

    # 4. 根据不同模式处理
    if args.is_sft:
        outputs = model(input_ids=input_ids, 
                       attention_mask=attention_mask, 
                       labels=labels, 
                       use_cache=False)
        return outputs.loss, None, None, None
    
    # 5. 非SFT模式的处理
    if args.stage == 2:
        teacher_logits, teacher_loss = get_teacher_outputs(
            teacher_engine, input_ids, attention_mask, labels, args)
        student_outputs = get_student_outputs(
            model, args, input_ids, labels, attention_mask)
        loss, kl_loss, student_ce_loss = compute_kl_loss(
            student_outputs, teacher_logits, labels, args,attention_mask=attention_mask)
    elif args.stage == 1:
        student_outputs = get_student_outputs(
            model, args, input_ids, labels, attention_mask)
        loss, kl_loss, student_ce_loss = get_attn_loss(
            args, student_outputs)
        teacher_loss = None
        
    return loss, teacher_loss, kl_loss, student_ce_loss
    
@time_function
def get_attn_loss(args, student_outputs):
    attn_from_wrapper = [student_outputs.attentions[i] for i in args.layers]
    # print(f'attn_from_wrapper {attn_from_wrapper}')
    loss = torch.stack(attn_from_wrapper, dim=0).mean()
    kl_loss = None
    student_cross_entropy_loss = None
    return loss,kl_loss,student_cross_entropy_loss

@time_function
def get_student_outputs(model, args, input_ids, labels, attention_mask):
    # print(f'student :attention_mask {attention_mask}')
    student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            labels=labels, use_cache=False, 
            output_attentions=args.stage==1)
        
    return student_outputs
@time_function
def get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args):
    # device = input_ids.device
    
    # # 将teacher模型移动到GPU
    # teacher_model.to(device)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=False)
    teacher_logits = teacher_outputs.logits
    teacher_loss = teacher_outputs.loss
    # 将teacher模型移回CPU
    # teacher_model.to('cpu')
    return teacher_logits,  teacher_loss

@time_function
def compute_kl_loss(student_outputs, teacher_logits, labels, args, attention_mask=None, chunk_size=4096):
    student_logits = student_outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    vocab_student = student_logits.shape[-1]
    vocab_teacher = teacher_logits.shape[-1]
    
    # Truncate teacher logits if necessary
    if vocab_teacher > vocab_student:
        teacher_logits = teacher_logits[:, :, :vocab_student]
    
    if args.enable_AKL:
        # For Adaptive KL loss, we'll modify the internal computation
        kl_loss = compute_adaptive_kl_loss(
            student_logits, 
            teacher_logits,
            attention_mask=attention_mask  # Pass attention mask to AKL function
        )
    else:
        # Compute softmax for student and teacher
        log_probs_student = F.log_softmax(student_logits, dim=-1)  # [batch_size, seq_len, vocab_size]
        targets = F.softmax(teacher_logits, dim=-1)    # [batch_size, seq_len, vocab_size]
        
        # Compute KL divergence without reduction
        kl_div_all = F.kl_div(
            log_probs_student,
            targets,
            reduction='none'  # Keep the full tensor to apply mask
        )  # [batch_size, seq_len, vocab_size]
        
        # Sum across vocabulary dimension first
        kl_div_per_token = kl_div_all.sum(dim=-1)  # [batch_size, seq_len]
        
        if attention_mask is not None:
            # Apply attention mask and compute mean only over attended positions
            masked_kl = kl_div_per_token * attention_mask
            kl_loss = masked_kl.sum() / (attention_mask.sum() + 1e-6)  # Add small epsilon for numerical stability
        else:
            # If no mask provided, take mean over all tokens
            kl_loss = kl_div_per_token.mean()
        
        del log_probs_student, targets, kl_div_all, kl_div_per_token
    
    # Get cross entropy loss from student outputs
    student_cross_entropy_loss = student_outputs.loss
    
    # Combine losses using weights from args
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    
    del student_logits, teacher_logits, labels
    if attention_mask is not None:
        del attention_mask
    return loss, kl_loss, student_cross_entropy_loss

import torch
import torch.nn.functional as F
import logging
import deepspeed
from typing import Dict, Any
class Stats:
    def __init__(self):
        self.total_calls = 0
        self.total_iterations = 0  # 总搜索迭代次数
        self.total_cutoff_sum = 0  # cutoff点位置总和
        self.total_samples = 0     # 总样本数
        self.cutoff_positions = []  # 存储每次找到的 cutoff 位置
        self.iteration_counts = []  # 存储每次迭代次数
        
stats = Stats()

@time_function
def find_cutoff_with_iterative_topk(probs: torch.Tensor, mu: float, k_top: int = 512) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    使用迭代式top-k查找cutoff点
    """
    batch_size, seq_len, vocab_size = probs.shape
    device = probs.device
    
    flat_probs = probs.view(-1, vocab_size)
    flat_batch_size = flat_probs.size(0)
    
    # DEBUG: 检查概率分布
    # if deepspeed.comm.get_rank() == 0 and torch.distributed.get_rank() == 0:
    #     sample_idx = 0
    #     top10_probs, _ = torch.topk(flat_probs[sample_idx], 10)
    #     logging.info(f"\nTop-10 probabilities sample: {top10_probs.cpu().tolist()}")
    #     logging.info(f"Sum of all probs: {flat_probs[sample_idx].sum():.4f}")
    #     logging.info(f"Number of probs > 0.01: {(flat_probs[sample_idx] > 0.01).sum().item()}")
    
    M = torch.zeros_like(flat_probs)
    processed_mask = torch.zeros_like(flat_probs, dtype=torch.bool)
    cumulative_sum = torch.zeros(flat_batch_size, device=device)
    cutoff_points = torch.zeros(flat_batch_size, dtype=torch.long, device=device)
    found_cutoff = torch.zeros(flat_batch_size, dtype=torch.bool, device=device)
    within_first_topk = torch.zeros(flat_batch_size, dtype=torch.bool, device=device)
    
    iteration = 0
    max_iterations = (vocab_size + k_top - 1) // k_top
    
    # 记录实际需要累积的token数量
    tokens_to_mu = torch.zeros(flat_batch_size, device=device)
    
    while not found_cutoff.all() and iteration < max_iterations:
        iteration += 1
        
        masked_probs = torch.where(processed_mask, torch.full_like(flat_probs, float('-inf')), flat_probs)
        curr_topk_probs, curr_topk_indices = torch.topk(masked_probs, min(k_top, vocab_size - k_top * (iteration-1)), dim=-1)
        
        processed_mask.scatter_(-1, curr_topk_indices, True)
        
        # 对每个样本进行累积概率计算
        for i in range(flat_batch_size):
            if found_cutoff[i]:
                continue
            
            # 计算当前chunk中的累积和，直到超过mu
            curr_cumsum = torch.cumsum(curr_topk_probs[i], dim=-1)
            needed_sum = mu - cumulative_sum[i]
            
            # 找到第一个累积和超过needed_sum的位置
            positions_over_mu = (curr_cumsum >= needed_sum).nonzero()
            
            if len(positions_over_mu) > 0:
                # 找到了cutoff点
                k = positions_over_mu[0].item() + 1  # +1是因为我们要包含这个位置
                cutoff_points[i] = k + k_top * (iteration-1)
                M[i].scatter_(-1, curr_topk_indices[i, :k], 1.0)
                found_cutoff[i] = True
                tokens_to_mu[i] = cutoff_points[i]
                
                if iteration == 1:
                    within_first_topk[i] = True
            else:
                # 没找到cutoff点，继续累积
                cumulative_sum[i] += curr_topk_probs[i].sum()
                M[i].scatter_(-1, curr_topk_indices[i], 1.0)
                tokens_to_mu[i] += curr_topk_probs[i].size(0)
    
    not_found_mask = ~found_cutoff
    if not_found_mask.any():
        cutoff_points[not_found_mask] = vocab_size
        tokens_to_mu[not_found_mask] = vocab_size
    
    # DEBUG: 输出统计信息
    # if deepspeed.comm.get_rank() == 0:
    #     avg_tokens_to_mu = tokens_to_mu.float().mean().item()
    #     max_tokens_to_mu = tokens_to_mu.max().item()
    #     logging.info(f"Average tokens needed to reach mu: {avg_tokens_to_mu:.2f}")
    #     logging.info(f"Max tokens needed to reach mu: {max_tokens_to_mu}")
        
    #     # 计算有多少样本在第一个chunk内找到了cutoff点
    #     first_chunk_ratio = (cutoff_points < k_top).float().mean().item()
    #     logging.info(f"Ratio of samples found in first chunk: {first_chunk_ratio:.2%}")
    
    M = M.view(batch_size, seq_len, vocab_size)
    cutoff_points = cutoff_points.view(batch_size, seq_len)
    within_first_topk = within_first_topk.view(batch_size, seq_len)
    
    return M, cutoff_points, within_first_topk, iteration

@time_function
def compute_adaptive_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    attention_mask: torch.Tensor = None,  # Added attention_mask parameter
    mu: float = 0.5,
    k_top: int = 512,
    debug: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Memory-optimized Adaptive KL Loss using iterative top-k search with attention mask support
    """
    global stats
    
    # Add numerical stability and apply softmax
    student_probs = F.softmax(student_logits, dim=-1).clamp(min=eps)
    teacher_probs = F.softmax(teacher_logits, dim=-1).clamp(min=eps)
    
    batch_size, seq_len, vocab_size = teacher_probs.shape
    device = teacher_probs.device

    # Get mask matrix M and other outputs
    M, cutoff_points, within_first_topk, iterations = find_cutoff_with_iterative_topk(
        teacher_probs, mu, k_top)
        
    # Update statistics
    stats.cutoff_positions.append(cutoff_points.float().mean().item())
    stats.iteration_counts.append(iterations)
    stats.total_calls += 1
    
    # Calculate gaps
    gaps = torch.abs(teacher_probs - student_probs)
    g_head = torch.sum(M * gaps, dim=-1)  # [batch_size, seq_len]
    g_tail = torch.sum((1 - M) * gaps, dim=-1)  # [batch_size, seq_len]
    
    total_gap = g_head + g_tail
    
    # Calculate weights with numerical stability
    w_head = torch.where(total_gap > eps, 
                        g_head / (total_gap + eps), 
                        0.5 * torch.ones_like(g_head))
    w_head = w_head.clamp(0.0, 1.0)
    w_tail = 1 - w_head
    
    # Calculate log probabilities
    student_log_probs = torch.log(student_probs.clamp(min=eps))
    teacher_log_probs = torch.log(teacher_probs.clamp(min=eps))
    
    # Calculate KL divergence components
    fkl_components = F.kl_div(
        student_log_probs.flatten(0, 1),
        teacher_probs.flatten(0, 1),
        reduction='none',
        log_target=False
    ).view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, vocab_size]
    
    rkl_components = F.kl_div(
        teacher_log_probs.flatten(0, 1),
        student_probs.flatten(0, 1),
        reduction='none',
        log_target=False
    ).view(batch_size, seq_len, -1)  # Reshape back to [batch_size, seq_len, vocab_size]
    
    # Sum over vocabulary dimension
    fkl = fkl_components.sum(-1).clamp(max=1e3)  # [batch_size, seq_len]
    rkl = rkl_components.sum(-1).clamp(max=1e3)  # [batch_size, seq_len]
    
    # Calculate weighted loss
    token_loss = w_head * fkl + w_tail * rkl  # [batch_size, seq_len]
    
    # Apply attention mask if provided
    if attention_mask is not None:
        token_loss = token_loss * attention_mask
        # Normalize by the number of attended tokens
        final_loss = token_loss.sum() / (attention_mask.sum() + eps)
    else:
        # If no mask provided, take mean over all tokens
        final_loss = token_loss.mean()
    
    # Handle NaN or Inf values
    if debug and (torch.isnan(final_loss).any() or torch.isinf(final_loss).any()):
        logging.error("\n=== Error in AKL Loss ===")
        logging.error(f"NaN or Inf detected in loss")
        logging.error(f"cutoff_points stats: mean={cutoff_points.float().mean():.2f}, min={cutoff_points.min().item()}, max={cutoff_points.max().item()}")
        logging.error(f"w_head stats: mean={w_head.mean():.4f}, min={w_head.min():.4f}, max={w_head.max():.4f}")
        logging.error(f"Loss components - fkl: mean={fkl.mean():.4f}, rkl: mean={rkl.mean():.4f}")
        logging.error(f"student_probs stats: mean={student_probs.mean():.4e}, min={student_probs.min():.4e}, max={student_probs.max():.4e}")
        logging.error(f"teacher_probs stats: mean={teacher_probs.mean():.4e}, min={teacher_probs.min():.4e}, max={teacher_probs.max():.4e}")
        if attention_mask is not None:
            logging.error(f"attention_mask stats: sum={attention_mask.sum():.4f}, mean={attention_mask.float().mean():.4f}")
        # Replace nan/inf values with 10
        final_loss = torch.tensor(10.0, device=device)
    
    # Log statistics periodically
    if stats.total_calls % 100 == 0 and deepspeed.comm.get_rank() == 0:
        avg_cutoff = sum(stats.cutoff_positions) / len(stats.cutoff_positions)
        avg_iterations = float(sum(stats.iteration_counts)) / len(stats.iteration_counts)
        logging.info(f"After {stats.total_calls} calls: Avg cutoff position = {avg_cutoff:.2f}, Avg iterations = {avg_iterations:.2f}, Current vocab_size = {vocab_size}")
        stats.cutoff_positions.clear()
        stats.iteration_counts.clear()
        
        # Log gaps with attention mask consideration
        if attention_mask is not None:
            masked_g_head = g_head * attention_mask
            masked_g_tail = g_tail * attention_mask
            mean_g_head = masked_g_head.sum() / (attention_mask.sum() + eps)
            mean_g_tail = masked_g_tail.sum() / (attention_mask.sum() + eps)
        else:
            mean_g_head = g_head.mean()
            mean_g_tail = g_tail.mean()
        
        logging.info(f"Mean Head gap: {mean_g_head:.4f}")
        logging.info(f"Mean Tail gap: {mean_g_tail:.4f}")

    return final_loss

def configure_optimizer(model, args):
    lr_decay = set()
    lr_1x = set()
    lr_2x = set()
    lr_3x = set()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (("_w1" in n) or ("_w2" in n)) and (args.layerwise_lr > 0):
            lr_1x.add(n)
        elif (("time_mix" in n) or ("time_maa" in n)) and (args.layerwise_lr > 0):
            lr_2x.add(n)
        elif (("time_decay" in n) or ("time_daaaa" in n)) and (args.layerwise_lr > 0):
            lr_2x.add(n)
        elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
            lr_1x.add(n)
        elif ("time_first" in n) and (args.layerwise_lr > 0):
            lr_3x.add(n)
        elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
            lr_decay.add(n)
        else:
            lr_1x.add(n)

    lr_decay = sorted(list(lr_decay))
    lr_1x = sorted(list(lr_1x))
    lr_2x = sorted(list(lr_2x))
    lr_3x = sorted(list(lr_3x))
    param_dict = {n: p for n, p in model.named_parameters()}
    
    if args.layerwise_lr > 0:
        optim_groups = [
                {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
            ]
    else:
        optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

    if args.weight_decay > 0:
        optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]

    if args.deepspeed:
        if args.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
        else:
            optimizer = FusedAdam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
    else:
        optimizer = Adam(optim_groups, lr=args.lr_init, betas=args.betas, eps=args.adam_eps)

    return optimizer
