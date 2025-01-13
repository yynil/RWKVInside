from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from transformers.modeling_utils import no_init_weights
from collections import OrderedDict

import os


with no_init_weights():
    model = AutoModelForCausalLM.from_pretrained("configs/ARWKV-7B", device_map="auto").bfloat16()
tokenizer = AutoTokenizer.from_pretrained("configs/ARWKV-7B")
device = "cuda"
prompt = "Give me a short introduction to large language model."

messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, do_sample=True)

generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

import os
os.system('clear')  # Windows 使用 'cls' 清除命令行
print(response)

