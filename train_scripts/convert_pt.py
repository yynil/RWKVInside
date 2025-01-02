import torch
import os
import gc
import argparse
from pathlib import Path

def convert_model(model_path, output_path):
    """Convert a single model file."""
    print(f'remove student_attn in {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')
    new_state_dict = {}
    replaced_key = 0
    
    for k, v in state_dict.items():
        if '.student_attn.' in k:
            new_key = k.replace('.student_attn.', '.time_mixer.')
            print(f'replace {k} with {new_key}')
            replaced_key += 1
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
            
    del state_dict
    gc.collect()
    
    print(f'save new model to {output_path} replaced {replaced_key} keys')
    torch.save(new_state_dict, output_path)

def process_directory(input_path, output_path):
    """Process all .bin files in a directory."""
    # 确保输出目录存在
    os.makedirs(output_path, exist_ok=True)
    
    # 如果输入是文件，直接处理
    if os.path.isfile(input_path):
        if input_path.endswith('.bin'):
            output_file = os.path.join(output_path, 
                                     os.path.basename(input_path).replace('.bin', '.pt'))
            convert_model(input_path, output_file)
        return
        
    # 如果输入是目录，处理所有.bin文件
    for file in os.listdir(input_path):
        if file.endswith('.bin'):
            input_file = os.path.join(input_path, file)
            output_file = os.path.join(output_path, file.replace('.bin', '.pt'))
            convert_model(input_file, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, 
                        help='path to the model file or directory containing .bin files')
    parser.add_argument('--output_path', type=str, 
                        help='output directory or file path')
    args = parser.parse_args()
    
    process_directory(args.model_path, args.output_path)