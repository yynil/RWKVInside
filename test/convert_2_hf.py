import os
import sys
import torch
# 添加项目根目录到sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(f"add {project_root} to sys.path")

from transformers.modeling_utils import no_init_weights


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='configs/qwen_7b.yaml')
    parser.add_argument("--ckpt_file", type=str, default=None)

    parser.add_argument('--wkv_has_norm', action='store_true', default=False)
    parser.add_argument("--wkv_version", type=int, default=7)
    parser.add_argument("--wkv_has_gate", action="store_true", default=False)
    parser.add_argument("--output_config_dir", type=str, default='configs/ARWKV-7B')
    args = parser.parse_args()
    # print(args)
    if args.wkv_version != 7:
        assert('not support yet')
    else:
        from hybrid_model_run_rwkv7 import create_rwkv_args, HybridModel
    from transformers import AutoModelForCausalLM, AutoTokenizer,AutoConfig
    config_file = args.config_file
    import yaml
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # print(config)
    model_id = config['Llama']['model_id']
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    transformer_config = AutoConfig.from_pretrained(model_id)
    # print(transformer_config)

    rwkv_args = create_rwkv_args(transformer_config, config)
    rwkv_args.has_group_norm = args.wkv_has_norm
    rwkv_args.gate_free = (not args.wkv_has_gate)
    model = HybridModel(rwkv_args,transformer_config)

    print(rwkv_args)

    ckpt_file = args.ckpt_file

    if ckpt_file is None:
        model.load_checkpoint(model_id)
    else:
        model.load_checkpoint(ckpt_file)
    dtype = torch.bfloat16
    model = model.to(dtype=dtype)
    

    state_dict = model.state_dict()
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("model.model."):  # 去掉多余的 "model."
            new_key = key.replace("model.model.", "model.")
        elif key.startswith("model.lm_head."):
            new_key = key.replace("model.", "")
        else:
            new_key = key
        new_state_dict[new_key] = value
    torch.save(new_state_dict, "hf_model_weights.pth")
    del state_dict, new_state_dict, model
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    import torch
    from transformers.modeling_utils import no_init_weights
    from collections import OrderedDict

    with no_init_weights():
        config = AutoConfig.from_pretrained(args.output_config_dir)
        model = AutoModelForCausalLM.from_config(config=config).bfloat16()
    print(model)
    
    state_dict = torch.load("hf_model_weights.pth", map_location="cpu", weights_only=True)  # 确保加载到 CPU
    model.load_state_dict(state_dict, strict=True)
    model.save_pretrained(args.output_config_dir)
    print("convert to huggingface models successfully!")