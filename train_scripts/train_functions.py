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
    input_ids = batch['input_ids']
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
        

    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).to(input_ids.device)

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
            student_outputs, teacher_logits, labels, args)
    elif args.stage == 1:
        student_outputs = get_student_outputs(
            model, args, input_ids, labels, attention_mask)
        loss, kl_loss, student_ce_loss = get_attn_loss(
            input_ids, student_outputs)
        teacher_loss = None
        
    return loss, teacher_loss, kl_loss, student_ce_loss
    
@time_function
def get_attn_loss(input_ids, student_outputs):
    loss = torch.stack(student_outputs.attentions, dim=0).mean()
    kl_loss = None
    student_cross_entropy_loss = None
    return loss,kl_loss,student_cross_entropy_loss

@time_function
def get_student_outputs(model, args, input_ids, labels, attention_mask):
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
def compute_kl_loss(student_outputs, teacher_logits, labels, args, chunk_size=4096):
    student_logits = student_outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    student_cross_entropy_loss = student_outputs.loss
    
    # 先对整个序列计算 softmax，保证归一化范围一致
    log_probs_student = F.log_softmax(student_logits, dim=-1)  # 在词表维度上做 softmax
    student_vocab_size = student_logits.size(-1)
    teacher_vocab_size = teacher_logits.size(-1)
    #if teacher_vocab_size > student_vocab_size, truncate teacher_logits brutally
    if teacher_vocab_size > student_vocab_size:
        if args.local_rank == 0:
            print(f"Truncating teacher logits from {teacher_vocab_size} to {student_vocab_size}")
        teacher_logits = teacher_logits[:, :, :student_vocab_size]
    else:
        if args.local_rank == 0:
            print(f"Padding teacher logits from {teacher_vocab_size} to {student_vocab_size}")
    targets = F.softmax(teacher_logits, dim=-1)
    
    kl_loss = F.kl_div(
            log_probs_student,
            targets,
            reduction='batchmean'
        )
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    del student_logits, teacher_logits, labels, log_probs_student, targets
    return loss, kl_loss, student_cross_entropy_loss


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
