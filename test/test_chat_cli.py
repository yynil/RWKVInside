import os
import sys
import torch
# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"add {project_root} to sys.path")
def creat_chatml(conversations):
    chatml = ""
    for conversation in conversations:
        chatml += f"<|im_start|>{conversation['role']}\n{conversation['content']}<|im_end|>\n"
    chatml += "<|im_start|>assistant\n"
    return chatml
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='configs/qwen_7b.yaml')
    parser.add_argument("--ckpt_file", type=str, default=None)
    parser.add_argument("--is_hybrid", action="store_true", default=False)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--has_norm', action='store_true', default=False)
    parser.add_argument("--is_rwkv_6", action="store_true", default=False)
    parser.add_argument("--gate_free", action="store_true", default=False)
    args = parser.parse_args()
    print(args)
    if args.is_rwkv_6:
        assert('not support yet')
    else:
        from hybrid_model_run_rwkv7 import create_rwkv_args, HybridModel
    from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
    config_file = args.config_file
    import yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_config = AutoConfig.from_pretrained(model_id)
    if args.is_hybrid:
        rwkv_args = create_rwkv_args(transformer_config, config)
        rwkv_args.has_group_norm = args.has_norm
        rwkv_args.gate_free = args.gate_free
        model = HybridModel(rwkv_args,transformer_config)
        ckpt_file = args.ckpt_file
        if ckpt_file is None:
            #try to load from model_id
            model.load_checkpoint(model_id)
        else:
            model.load_checkpoint(ckpt_file)
        dtype = torch.bfloat16
        model = model.to(dtype=dtype)
        num_gpus = args.num_gpus
        if num_gpus > 1:
            num_layers = model.model.config.num_hidden_layers
            device_map = {}
            average_layers = num_layers // num_gpus
            for i in range(num_layers):
                device_map[f'model.layers.{i}'] = i // average_layers
            device_map['model.embed_tokens'] = 'cpu'
            device_map['model.norm'] = 'cpu'
            device_map['model.rotary_emb'] = 'cpu'
            device_map['lm_head'] = 'cpu'
            from accelerate import dispatch_model
            model.model = dispatch_model(model.model, device_map=device_map,offload_buffers=True)
        else:
            model = model.to(f'cuda:0')
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(torch.bfloat16)
        model = model.to('cuda:0')
    print(model)
    input('press any key to continue')
    from transformers import GenerationConfig
    gen_config = GenerationConfig(
        max_new_tokens=1024,
        stop_strings = ["<|im_end|>"],
        do_sample = True,
        use_cache = True,
        temperature = 0.7,
        top_k = 10,
        top_p = 0.9,
        min_p = 0.05,
        repetition_penalty = 1.1,
    )
    message = input('please input message:')
    conversation = [{
        'role': 'user',
        'content': message
    }]
    from utilities import HybridCache
    cache = HybridCache()
    while True:
        current_input_text = creat_chatml(conversation)
        print(current_input_text)
        input_ids = tokenizer(current_input_text, return_tensors="pt").to("cuda:0")
        print(input_ids)
        input_length = input_ids.input_ids.shape[1]
        print(f'input length: {input_length}')
        with torch.no_grad():
            if args.is_hybrid:
                print("use hybrid model to generate")
                model_to_use = model.model
            else:
                model_to_use = model
            print(model_to_use)
            output = model_to_use.generate(
                    input_ids=input_ids['input_ids'],
                    attention_mask=input_ids['attention_mask'],
                    past_key_values=cache,
                    generation_config=gen_config,
                    tokenizer = tokenizer,
                    use_cache = True,
                )
        print(f'generated {output.shape[1] - input_length} tokens')
        generated_text = tokenizer.decode(output[0,input_length:], skip_special_tokens=True)            
        print(generated_text)
        conversation.append({
            'role': 'assistant',
            'content': generated_text
        })
        message = input('enter message:')
        if message == 'exit':
            break
        if message == 'clear':
            conversation = []
            cache = HybridCache()
            message = input('enter message:')
        conversation.append({
            'role': 'user',
            'content': message
        })
        