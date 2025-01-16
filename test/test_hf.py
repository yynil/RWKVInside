from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
from transformers.modeling_utils import no_init_weights
from collections import OrderedDict

import os
from transformers import TextIteratorStreamer
import threading
import sys
model_path = sys.argv[1]

device = "cuda:2"
with no_init_weights():
    model = AutoModelForCausalLM.from_pretrained(model_path,device_map="cpu").half()
    model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
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

os.system('clear')  # Windows 使用 'cls' 清除命令行
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# 在单独的线程中生成文本
generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=512, do_sample=True)
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 逐步输出生成的文本
print("Streaming output:")
for new_text in streamer:
    print(new_text, end="", flush=True)

# 等待生成线程完成
thread.join()

