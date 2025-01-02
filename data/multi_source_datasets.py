from datasets import load_dataset,interleave_datasets,concatenate_datasets
import glob
import pandas as pd
import os
import torch
def load_parquet_dataset(path,split='train', val_size=0.01):
    files = glob.glob(path + '*.parquet') + glob.glob(path + '**/*.parquet')
    print(f'加载 parquet 数据集，文件：{files}')
    dataset = load_dataset('parquet', data_files=files)
    print(f'数据集：{dataset}')
    
    # 分割数据集
    if split == 'train':
        dataset = dataset['train'].train_test_split(test_size=val_size)
        train_dataset = post_process_dataset(dataset['train'], 'train')
        val_dataset = post_process_dataset(dataset['test'], 'validation')
        return train_dataset, val_dataset
    else:
        return post_process_dataset(dataset[split], split)

def load_json_gz_dataset(path, split='train', val_size=0.01):
    files = glob.glob(path + '*.jsonl.gz') + glob.glob(path + '**/*.jsonl.gz')
    print(f'加载 jsonl.gz 数据集，文件：{files}')
    dataset = load_dataset('json', data_files=files)
    print(f'数据集：{dataset}')
    
    # 分割数据集
    if split == 'train':
        dataset = dataset['train'].train_test_split(test_size=val_size)
        print(f'分割后的数据集：{dataset}')
        train_dataset = post_process_dataset(dataset['train'], 'train')
        val_dataset = post_process_dataset(dataset['test'], 'validation')
        return train_dataset, val_dataset
    else:
        return post_process_dataset(dataset['train'], split)

def load_jsonl_dataset(path,split='train',val_size=0.01):
    files = glob.glob(path + '**/*.jsonl')+glob.glob(path+'*.jsonl')
    print(f'loading jsonl dataset from {files}')
    dataset = load_dataset('json', data_files=files)
    if split == 'train':
        dataset = dataset['train'].train_test_split(test_size=val_size)
        train_dataset = post_process_dataset(dataset['train'], 'train')
        val_dataset = post_process_dataset(dataset['test'], 'validation')
        return train_dataset, val_dataset
    else:
        dataset = load_dataset('json', data_files=files)
        dataset = post_process_dataset(dataset,split)
    return dataset

def check_dataset_type(path):
    parquet_files = glob.glob(path + '**/*.parquet')+glob.glob(path+'*.parquet')
    json_gz_files = glob.glob(path + '**/*.jsonl.gz')+glob.glob(path+'*.jsonl.gz')
    jsonl_files = glob.glob(path + '**/*.jsonl')+glob.glob(path+'*.jsonl')
    types = []
    if parquet_files and len(parquet_files) > 0:
        types.append('parquet')
    if json_gz_files and len(json_gz_files) > 0 :
        types.append('json_gz')
    if jsonl_files and len(jsonl_files) > 0:
        types.append('jsonl')
    return types

def post_process_dataset(dataset, split):
    # 只保留 'text' 列
    dataset = dataset.map(lambda x: {'text': x['text']}, batched=True, batch_size=1000, num_proc=10, remove_columns=dataset.column_names)
    return dataset

def jsonl_to_parquet(jsonl_path, output_dir, file_extension):
    # 读取 JSONL 文件
    dataset = load_dataset('json', data_files=jsonl_path, split='train')
    
    # 只保留 'text' 列
    dataset = dataset.map(lambda x: {'text': x['text']}, batched=True, batch_size=1000, num_proc=10, remove_columns=dataset.column_names)
    
    # 转换为 Pandas DataFrame
    df = pd.DataFrame(dataset)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成 Parquet 文件路径
    parquet_file = os.path.join(output_dir, os.path.basename(jsonl_path).replace(file_extension, '.parquet'))
    
    # 保存为 Parquet 文件
    df.to_parquet(parquet_file, compression='snappy')
    print(f'已将 {jsonl_path} 转换为 {parquet_file}')

def load_and_interleave_datasets(paths, split='train', val_size=0.01):
    train_datasets = []
    val_datasets = []
    for path in paths:
        data_types = check_dataset_type(path)
        print(f'数据集类型：{data_types}，路径：{path}')
        if 'parquet' in data_types:
            train_dataset, val_dataset = load_parquet_dataset(path, split=split, val_size=val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        if 'json_gz' in data_types:
            train_dataset, val_dataset   = load_json_gz_dataset(path,split=split,val_size=val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        if 'jsonl' in data_types:
            train_dataset, val_dataset = load_jsonl_dataset(path,split=split,val_size=val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
    train_dataset = concatenate_datasets(train_datasets)
    val_dataset = concatenate_datasets(val_datasets)
    return train_dataset, val_dataset

def tokenize_dataset(dataset, tokenizer, max_seq_length):
    def tokenize_function(examples):
        batch = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=max_seq_length, return_tensors="pt")
        input_ids, labels = batch["input_ids"], batch["input_ids"].clone()
        
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = tokenizer.eos_token_id
        
        padding_mask = input_ids.eq(tokenizer.eos_token_id)
        input_ids.masked_fill_(padding_mask, tokenizer.pad_token_id)
        labels.masked_fill_(padding_mask, -100)
        
        return {"input_ids": input_ids.tolist(), "labels": labels.tolist()}
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=10, remove_columns=['text'])
    return tokenized_dataset
def data_collator_with_pad(features,max_seq_length,pad_token_id):
    input_ids = [f["input_ids"][0:max_seq_length] 
                 if len(f['input_ids']) >= max_seq_length 
                 else 
                 f["input_ids"]+[pad_token_id]*(max_seq_length-len(f['input_ids']))
                 for f in features]
    labels = [f["labels"][0:max_seq_length] 
              if len(f['labels']) >= max_seq_length 
              else
                f["labels"]+[-100]*(max_seq_length-len(f['labels']))
              for f in features]
    return {"input_ids": torch.tensor(input_ids,dtype=torch.long), "labels": torch.tensor(labels,dtype=torch.long)}

def data_collator(features,max_seq_length):
    input_ids = [f["input_ids"][0:max_seq_length] for f in features]
    labels = [f["labels"][0:max_seq_length] for f in features]
    return {"input_ids": torch.tensor(input_ids,dtype=torch.long), "labels": torch.tensor(labels,dtype=torch.long)}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs='+', required=True)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--convert_to_parquet', action='store_true', help='是否将 JSONL 文件转换为 Parquet')
    parser.add_argument('--is_save_to_disk', action='store_true', help='是否将数据集保存到磁盘')
    parser.add_argument('--output_dirs', type=str, nargs='+', help='转换后的 Parquet 文件输出目录')
    parser.add_argument('--file_extension', type=str, choices=['.jsonl', '.jsonl.gz'], help='要转换的文件扩展名')
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--max_seq_length', type=int, default=2048)

    args = parser.parse_args()
    
    if args.convert_to_parquet:
        if not args.output_dirs or len(args.output_dirs) != len(args.paths):
            raise ValueError("必须提供与输入路径数量相同的输出目录 --output_dirs")
        if not args.file_extension:
            raise ValueError("必须提供文件扩展名 --file_extension")
        for path, output_dir in zip(args.paths, args.output_dirs):
            jsonl_files = glob.glob(path + '**/*' + args.file_extension) + glob.glob(path + '*' + args.file_extension)
            for jsonl_file in jsonl_files:
                jsonl_to_parquet(jsonl_file, output_dir, args.file_extension)
    else:
        train_datasets = []
        val_datasets = []
        for path in args.paths:
            train_dataset, val_dataset = load_and_interleave_datasets([path], split='train', val_size=args.val_size)
            train_datasets.append(train_dataset)
            val_datasets.append(val_dataset)
        train_dataset = concatenate_datasets(train_datasets)
        val_dataset = concatenate_datasets(val_datasets)
        print(f'训练集：{train_dataset}')
        print(f'验证集：{val_dataset}')
        print("训练集样本：", train_dataset[100])
        print('训练样本:', train_dataset[101])
        print('训练样本:', val_dataset[102])
        print("验证样本：", val_dataset[100])
        print('验证样本:', val_dataset[101])
        print('验证样本:', val_dataset[102])
        print(f'训练集长度：{len(train_dataset)}')
        print(f'验证集长度：{len(val_dataset)}')
        if args.is_save_to_disk:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_id)
            tokenizer.pad_token = tokenizer.eos_token
            os.makedirs(args.output_dirs[0], exist_ok=True)
            os.makedirs(args.output_dirs[1], exist_ok=True)
            train_dataset = tokenize_dataset(train_dataset, tokenizer, max_seq_length=args.max_seq_length)
            val_dataset = tokenize_dataset(val_dataset, tokenizer, max_seq_length=args.max_seq_length)
            train_dataset.save_to_disk(args.output_dirs[0])
            val_dataset.save_to_disk(args.output_dirs[1])
