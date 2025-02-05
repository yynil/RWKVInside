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
prompt = "计算 f(x) =log_10(5x^2+7x) 在 x = 2 处的导数。"
system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process are enclosed within <think> </think>, i.e., <think> reasoning process here </think>**Final Answer:**\nanswer here. "

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
text = text + "<think>"
print(text)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
print(model_inputs)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

# 在单独的线程中生成文本
generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=2048, do_sample=True,tokenizer=tokenizer,stop_strings=["<｜end▁of▁sentence｜>"])
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 逐步输出生成的文本
print("Streaming output:")
for new_text in streamer:
    print(new_text, end="", flush=True)

# 等待生成线程完成
thread.join()

