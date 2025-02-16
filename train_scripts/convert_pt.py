import torch
import os
import gc
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM

def convert_model(model_path, output_path,copy_mlp_from_original,original_model):
    """Convert a single model file."""
    print(f'remove student_attn in {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {}
    replaced_key = 0
    mlp_layers = []
    is_gate_exists = False
    #check if the name with  "g1" "g2" suffix exists 
    for k, v in state_dict.items():
        if '.student_attn.' in k:
            if ".key.weight" in k:
                #model.model.layers.18.self_attn.student_attn.key.weight
                mlp_layers.append(int(k.split('.')[3]))
            new_key = k.replace('.student_attn.', '.time_mixer.')
            print(f'replace {k} with {new_key}')
            replaced_key += 1
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
        if k.endswith('g1') or k.endswith('g2'):
            is_gate_exists = True
            
    if not is_gate_exists:
        print(f'remove all params ends with x_g')
        for k, v in state_dict.items():
            if k.endswith('x_g'):
                print(f'remove {k}')
                new_key = k.replace('.student_attn.', '.time_mixer.')
                replaced_key += 1
                del new_state_dict[new_key]
    del state_dict
    gc.collect()
    if copy_mlp_from_original:
        for i in mlp_layers:
            original_mlp = original_model.model.layers[i].mlp
            #iterate over the original mlp layers
            for k, v in original_mlp.state_dict().items():
                new_state_dict[f"model.model.layers.{i}.mlp.{k}"] = v
                print(f'copy mlp layer {k} from original model to model.model.layers.{i}.mlp.{k}')
    print(f'save new model to {output_path} replaced {replaced_key} keys')
    torch.save(new_state_dict, output_path)

def process_directory(input_path, output_path, original_model_path, copy_mlp_from_original):
    """Process all .bin files in a directory."""
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    if copy_mlp_from_original:
        print(f'copy mlp from {original_model_path}')
        original_model = AutoModelForCausalLM.from_pretrained(original_model_path)
    else:
        original_model = None
    
    # 如果输入是文件，直接处理
    if os.path.isfile(input_path):
        if input_path.endswith('.bin'):
            output_file = os.path.join(output_path, 
                                     os.path.basename(input_path).replace('.bin', '.pt'))
            convert_model(input_path, output_file,copy_mlp_from_original,original_model)
        return
        
    # 如果输入是目录，处理所有.bin文件
    for file in os.listdir(input_path):
        if file.endswith('.bin'):
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file.replace('.bin', '.pt'))
            convert_model(input_file, output_file,copy_mlp_from_original,original_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        help='path to the model file or directory containing .bin files')
    parser.add_argument('--output_path', type=str, 
                        help='output directory or file path')
    parser.add_argument('--original_model_path', type=str,help='path to the original qwen model')
    parser.add_argument('--copy_mlp_from_original', action='store_true',default=False,help='whether to copy the mlp layer from the original model')
    args = parser.parse_args()
    
    process_directory(args.model_path, args.output_path, args.original_model_path, args.copy_mlp_from_original)