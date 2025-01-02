from dataclasses import dataclass
from typing import Dict, List, Union, Optional
import torch
from transformers import PreTrainedTokenizer
import datasets
import glob
import os

def detect_format(directory) -> str:
    """
    Detect the format of the dataset in the given directory.
    Args:
    directory: The directory containing the dataset files.
    Returns:
    The format of the dataset.['jsonl', 'parquet', 'huggingface','unknown']
    """
    #if dataset_info.json exists, it is a huggingface dataset
    if os.path.exists(os.path.join(directory, 'dataset_info.json')):
        return 'huggingface'
    #if parquet files exist, it is a parquet dataset
    if glob.glob(os.path.join(directory, '*.parquet')):
        return 'parquet'
    #if jsonl files exist, it is a jsonl dataset
    if glob.glob(os.path.join(directory, '*.jsonl')):
        return 'jsonl'
    return 'unknown'
def load_jsonl_dataset(file_path):
    jsonl_files = glob.glob(file_path+"/*.jsonl")
    print(f'load jsonl files: {jsonl_files}')
    dataset = datasets.load_dataset('json', data_files=jsonl_files)['train']
    return dataset

def load_parquet_dataset(file_path):
    parquet_files = glob.glob(file_path+"/*.parquet")
    print(f'load parquet files: {parquet_files}')
    dataset = datasets.load_dataset('parquet', data_files=parquet_files)['train']
    return dataset

def load_huggingface_dataset(file_path):
    dataset = datasets.load_from_disk(file_path)
    return dataset

def check_feature(dataset):
    """
    check the features of dataset
    if the feature contains messages and the value is a list, return "conversation"
    if the feature contains text and the value is a string, return "text"
    """
    for feature, value in dataset[0].items():
        if feature == 'messages' and isinstance(value, list):
            return 'conversation'
        if feature == 'text' and isinstance(value, str):
            return 'text'
    return 'unknown'

load_functions = {
    'jsonl': load_jsonl_dataset,
    'parquet': load_parquet_dataset,
    'huggingface': load_huggingface_dataset
}

def convert_conversation_to_text(example: Dict) -> Dict:
    """
    Convert a single conversation to ChatML format.
    
    Args:
        example: A dictionary containing a 'messages' list of conversations
        
    Returns:
        A dictionary with a single 'text' key containing the ChatML formatted conversation
    """
    result = []
    for message in example['messages']:
        result.extend([
            f"<|im_start|>{message['role']}\n",
            f"{message['content']}\n",
            "<|im_end|>\n"
        ])
    
    return {'text': "".join(result)}

def convert_conversational_ds_to_text(ds: datasets.Dataset) -> datasets.Dataset:
    """
    Convert a conversational dataset to ChatML format.
    
    Args:
        ds: A dataset containing 'messages' lists of conversations
        
    Returns:
        A dataset with a single 'text' key containing the ChatML formatted conversation
    """
    return ds.map(convert_conversation_to_text,num_proc=8,  # 使用8个进程并行处理
        remove_columns=ds.column_names,  # 移除所有原始列
        desc="Converting conversations"  # 显示进度条描述
    )



def load_datasets_from_directories(directories):
    """
    Load datasets from directories.
    Args:
    directories: A list of directories containing the dataset files.
    Returns:
    A list of datasets.
    """
    all_ds = []
    for directory in directories:
        dataset_type = detect_format(directory)
        print(f"Detected dataset type: {dataset_type}")
        if dataset_type == 'unknown':
            print(f"Unknown dataset type for directory: {directory}")
            continue
        ds = load_functions[dataset_type](directory)
        feature_type = check_feature(ds)
        if feature_type == 'conversation':
            ds = convert_conversational_ds_to_text(ds)
        else:
            ds = ds.select_columns(['text'])
        print(f"Loaded dataset from directory: {directory}")
        all_ds.append(ds)
    return all_ds

@dataclass
class StreamingCLMDataCollator:
    """
    Data collator that handles streaming tokenization for causal language modeling
    with next token prediction.
    
    Args:
        tokenizer: The tokenizer to use for tokenization
        max_length: Maximum sequence length
        pad_to_multiple_of: Optional length to pad sequences to a multiple of
    """
    tokenizer: PreTrainedTokenizer
    max_length: int
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, examples: List[Dict[str, Union[str, List[str]]]]) -> Dict[str, torch.Tensor]:
        """
        Tokenize and collate examples into a batch, shifting labels for next token prediction.
        
        Args:
            examples: List of examples with 'text' field
            
        Returns:
            Batch dictionary with input_ids, attention_mask, and labels
        """
        # Extract texts from examples, handling both string and list inputs
        texts = [
            ex['text'] if isinstance(ex['text'], str) else ex['text'][0]
            for ex in examples
        ]
        
        # Tokenize all texts in the batch
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Get the input IDs and attention mask
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        
        # Create labels for next token prediction
        # Labels at position i should be the token at position i+1
        labels = input_ids.clone()
        # Move tokens one position left: [t1, t2, t3, t4, pad] -> [t2, t3, t4, pad, pad]
        labels[:, :-1] = input_ids[:, 1:]
        # Set the last position to padding token or -100
        last_token = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -100
        labels[:, -1] = last_token
        
        # Set labels to -100 where we have padding in inputs
        labels[attention_mask == 0] = -100
        
        # Also set label to -100 for the position before padding starts
        # This ensures we don't predict padding tokens
        padding_start = attention_mask.sum(dim=1) - 1  # Get the last non-padding position
        for i in range(len(padding_start)):
            if padding_start[i] > 0:  # Only if there is padding
                labels[i, padding_start[i]] = -100
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }       
if __name__ == '__main__':
    directory = '/home/yueyulin/data/finemath/finemath-4plus/'
    dataset_type = detect_format(directory)
    print(f"Detected dataset type: {dataset_type}")
    dataset = load_functions[dataset_type](directory)
    print(dataset)
    #find unique value of language
    print(dataset.unique('language'))
    print(dataset[0]['text'])
    print(check_feature(dataset))
    dataset = dataset.select_columns(['text'])
    print(dataset[0]['text'])
    print(dataset)
    
    directory = '/home/yueyulin/data/Mobius/standard/'
    dataset_type = detect_format(directory)
    print(f"Detected dataset type: {dataset_type}")
    dataset = load_functions[dataset_type](directory)
    print(dataset)
    print(check_feature(dataset))
    dataset = convert_conversational_ds_to_text(dataset)
    print(dataset)
    print(check_feature(dataset))
    print(dataset[0]['text'])
    directories = ['/home/yueyulin/data/finemath/finemath-4plus/', '/home/yueyulin/data/Mobius/standard/']
    
    all_ds = load_datasets_from_directories(directories)
    print(all_ds)
    con_ds = datasets.concatenate_datasets(all_ds)
    print(con_ds)
    print(con_ds[0]['text'])
    model_path = '/home/yueyulin/model/qwen_7b_stage3_4k_splits/'
    from transformers import DataCollatorForLanguageModeling,AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = StreamingCLMDataCollator(tokenizer=tokenizer, max_length=4096)
    import torch
    from torch.utils.data import DataLoader
    data_loader = DataLoader(con_ds, batch_size=1, collate_fn=data_collator)
    for batch in data_loader:
        print(batch)
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['labels'].shape)
        break