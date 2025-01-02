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
from server.nccl_client import InferenceClient
from profiler import time_function

def rank0_print(*args, **kwargs):
    if deepspeed.comm.get_rank() == 0:
        print(*args, **kwargs)
        
def initialize_nccl_client(args):
    if not args.is_sft and args.teacher_client_mode:
        
        rank = torch.cuda.current_device()
        logging.info(f'开始初始化NCCL客户端: rank {rank}')
        # world_size = args.world_size
        cp.cuda.Device(rank).use()
        # Create student client
        groups = args.groups
        for group in groups:
            if rank in group['cuda_devices']:
                world_size = group['num_groups']
                global_rank = group['global_ranks'][group['cuda_devices'].index(rank)]
                batch_size = args.micro_bsz
                num_hidden_layers = args.n_layer
                hidden_size = args.n_embd
                max_length = args.max_seq_length
                vocab_size = args.vocab_size
                nccl_file = group['nccl_file']
                break
        with open(nccl_file,'r') as f:
            import json 
            nccl_id = json.load(f)['nccl_ids']
            nccl_id = tuple(nccl_id)
        import os
        process_id = os.getpid()
        print(f'PID:{process_id} rank {rank} is initializing student client, world_size is {world_size}, global_rank is {global_rank} with nccl_id {nccl_id}')
        client = InferenceClient(
            world_size = world_size,
            global_rank = global_rank,
            local_rank = rank,
            nccl_id = nccl_id,
            batch_size = batch_size,
            length = max_length,
            vocab_size = vocab_size,
            num_layers = num_hidden_layers,
            hidden_size = hidden_size,
            output_hidden_states = args.is_hidden_align
        )

        logging.info('NCCL客户端初始化完成')
        return client
    
@time_function
def train_step_vl(model, batch, args, teacher_engine=None, tokenizer=None):
    images = batch['images']
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    image_sizes = batch['image_sizes']

    if not args.is_sft:
        teacher_logits, teacher_hidden_states, teacher_loss = get_teacher_outputs_vl(teacher_engine, input_ids, attention_mask,labels,image_sizes,images, args)
        
        student_outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels,images=images,image_sizes=image_sizes, use_cache=False, output_hidden_states=args.is_hidden_align)
        
        if not args.is_hidden_align:
            loss, kl_loss, student_cross_entropy_loss = compute_kl_loss(student_outputs, teacher_logits, labels, args)
        else:
            kl_loss = None
            student_cross_entropy_loss = None
            loss = compute_hidden_state_loss(student_outputs, teacher_hidden_states)
        
        return loss, teacher_loss, kl_loss, student_cross_entropy_loss
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,images=images,image_sizes=image_sizes, use_cache=False)
        return outputs.loss, None, None, None
@time_function
def get_teacher_outputs_vl(teacher_model, 
                           input_ids, 
                           attention_mask, 
                           labels, 
                           image_sizes,
                           images,
                           args):
    # device = input_ids.device
    
    # # 将teacher模型移动到GPU
    # teacher_model.to(device)
    
    with torch.no_grad():
        teacher_outputs = teacher_model.forward(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    images=images,
                                    labels=labels,
                                    image_sizes=image_sizes,
                                    use_cache=False,
                                    output_hidden_states=args.is_hidden_align)
    
    teacher_logits = teacher_outputs.logits
    teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None
    teacher_loss = teacher_outputs.loss
    if teacher_hidden_states is not None:
        teacher_hidden_states = torch.cat(teacher_hidden_states, dim=0)
    # 将teacher模型移回CPU
    # teacher_model.to('cpu')
    return teacher_logits, teacher_hidden_states, teacher_loss

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
    labels = batch['labels']
    # 处理 labels 不存在的情况
    if 'labels' not in batch:
        # 创建左移的 labels: 把 input_ids 往右补充一个 pad token,然后去掉最后一个 token
        labels = torch.cat([input_ids[:, 1:], 
                          torch.full((input_ids.shape[0], 1), 
                                   tokenizer.pad_token_id, 
                                   device=input_ids.device)], dim=1)
    else:
        labels = batch['labels']
        
    # 检查 labels 是否已经左移
    # 通过比较第一个非pad位置的 token 是否相同来判断
    first_nonpad_pos = (input_ids != tokenizer.pad_token_id).nonzero()[:, 1][0]
    if input_ids[0, first_nonpad_pos] == labels[0, first_nonpad_pos]:
        # labels 没有左移,需要左移 1 位
        labels = torch.cat([labels[:, 1:], 
                          torch.full((labels.shape[0], 1), 
                                   tokenizer.pad_token_id, 
                                   device=labels.device)], dim=1)
    attention_mask = torch.ne(input_ids, tokenizer.pad_token_id).to(input_ids.device)

    if not args.is_sft:
        
        if args.stage == 2:
            # rank0_print(f'calculate loss for stage 2, input_ids shape is {input_ids.shape}')
            teacher_logits, teacher_hidden_states, teacher_loss = get_teacher_outputs(teacher_engine, input_ids, attention_mask, labels, args)
        
        student_outputs = get_student_outputs(model, args, input_ids, labels, attention_mask)
        
        if args.stage == 2:
            # rank0_print(f'calculate loss for stage 2, input_ids shape is {input_ids.shape}')
            loss, kl_loss, student_cross_entropy_loss = compute_kl_loss(student_outputs, teacher_logits, labels, args)
        else:
            loss, kl_loss, student_cross_entropy_loss = get_attn_loss(input_ids, student_outputs)
            # loss = compute_hidden_state_loss(student_outputs, teacher_hidden_states)
            teacher_loss = None
        return loss, teacher_loss, kl_loss, student_cross_entropy_loss
    else:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        return outputs.loss, None, None, None
    
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
def get_teacher_outputs_client_mode(model, input_ids, args):
    b, t = input_ids.shape
    logging.info(f'rank {args.local_rank} is sending input_ids to server, shape is {input_ids.shape}')
    result = model.client.forward(input_ids=input_ids)
    if args.is_hidden_align:
        logits, hidden_states = result
        return logits, hidden_states
    else:
        logits = result
        return logits,None
@time_function
def get_teacher_outputs(teacher_model, input_ids, attention_mask, labels, args):
    # device = input_ids.device
    
    # # 将teacher模型移动到GPU
    # teacher_model.to(device)
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
    teacher_logits = teacher_outputs.logits
    teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None
    teacher_loss = teacher_outputs.loss
    if teacher_hidden_states is not None:
        teacher_hidden_states = torch.cat(teacher_hidden_states, dim=0)
    # 将teacher模型移回CPU
    # teacher_model.to('cpu')
    return teacher_logits, teacher_hidden_states, teacher_loss

@time_function
def compute_kl_loss(student_outputs, teacher_logits, labels, args, chunk_size=4096):
    student_logits = student_outputs.logits  # shape: [batch_size, seq_len, vocab_size]
    student_cross_entropy_loss = student_outputs.loss
    total_length = student_logits.size(1)
    
    # 先对整个序列计算 softmax，保证归一化范围一致
    log_probs_student = F.log_softmax(student_logits, dim=-1)  # 在词表维度上做 softmax
    targets = F.softmax(teacher_logits, dim=-1)
    
    if total_length <= chunk_size:
        kl_loss = F.kl_div(
            log_probs_student,
            targets,
            reduction='batchmean'
        )
    else:
        # 分段计算总的 KL divergence
        total_kl_div = 0
        chunk_size = 256
        num_chunks = (total_length + chunk_size - 1) // chunk_size
        
        for chunk_start in range(0, total_length, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_length)
            # 取出已经计算好的 log_probs 和 targets
            chunk_log_probs = log_probs_student[:, chunk_start:chunk_end, :]
            chunk_targets = targets[:, chunk_start:chunk_end, :]
            
            # 计算当前段的 KL div 并累加
            chunk_kl_div = F.kl_div(
                chunk_log_probs,
                chunk_targets,
                reduction='none'  # 先不做 reduction
            )
            total_kl_div += chunk_kl_div.sum()  # 累加所有元素
            
            # 释放临时变量
            del chunk_log_probs, chunk_targets
        
        # 最后统一做归一化，保证和整体计算结果一致
        kl_loss = total_kl_div / (student_logits.size(0) * total_length * student_logits.size(2))
    
    loss = args.kl_weight * kl_loss + args.ce_weight * student_cross_entropy_loss
    del student_logits, teacher_logits, labels, log_probs_student, targets
    return loss, kl_loss, student_cross_entropy_loss

@time_function
def compute_hidden_state_loss(student_outputs, teacher_hidden_states):
    # mask = torch.ne(labels, -100).to(labels.device)
    # mask = mask.unsqueeze(1).unsqueeze(3)
    student_hidden_states = torch.cat(student_outputs.hidden_states, dim=0)
    # student_hidden_states = student_hidden_states * mask
    # teacher_hidden_states = teacher_hidden_states * mask
    diff = student_hidden_states - teacher_hidden_states
    norms = torch.linalg.vector_norm(diff, dim=-1)
    scaled_norms = norms * (student_hidden_states[0].size(-1) ** -0.5)
    loss = scaled_norms.mean()
    # loss = torch.linalg.vector_norm(teacher_hidden_states - student_hidden_states,dim=-1).mean()*(teacher_hidden_states[0].size(-1)**-0.5)
    # loss = F.mse_loss(student_hidden_states, teacher_hidden_states.to(student_hidden_states.dtype))
    return loss

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

def validation_step(model, batch, args, teacher_model=None, tokenizer=None):
    input_ids = batch['input_ids']
    labels = batch['labels']
    attention_mask = torch.ne(input_ids, tokenizer.eos_token_id).to(input_ids.device)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    loss = outputs.loss
    
    # 计算perplexity
    perplexity = torch.exp(loss)
    
    result = {'val_loss': loss, 'val_perplexity': perplexity}
    
    if not args.is_sft:
        if args.teacher_client_mode:
            teacher_logits, teacher_hidden_states = get_teacher_outputs_client_mode(model, input_ids, args)
        else:
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False, output_hidden_states=args.is_hidden_align)
            teacher_logits = teacher_outputs.logits
            teacher_hidden_states = teacher_outputs.hidden_states if args.is_hidden_align else None

        # 计算teacher's loss和perplexity
        teacher_logits_reshaped = teacher_logits.view(-1, teacher_logits.size(-1))
        labels_reshaped = labels.view(-1)
        teacher_loss = F.cross_entropy(teacher_logits_reshaped, labels_reshaped)
        teacher_perplexity = torch.exp(teacher_loss)

        result.update({
            'val_teacher_loss': teacher_loss,
            'val_teacher_perplexity': teacher_perplexity
        })

    return result
