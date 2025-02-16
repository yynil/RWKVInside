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
prompt = "Joan picked some apples from the orchard, and gave 27 apples to Melanie. Joan has 16 apples now. How many apples did Joan pick from the orchard?"
prompt = "There were 27 boys and 35 girls on the playground at recess. There were _____ children on the playground at recess."
prompt = "What is the smallest positive perfect cube thatcan be written as the sum of three consecutive integers?"
prompt = "计算一下1+23445-8的结果"
# prompt = "请用感性的、文学性的文字回答天空为什么是蓝色的？"
# prompt = "The world's largest rainforest, home to approximately three million species of plants and animals, is named after which river?"
system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process are enclosed within <think> </think>, i.e., <think> reasoning process here </think>**Final Answer:**\nanswer here. "

system_prompt = """DeepSeek-R1 System Prompt
You are DeepSeek-R1, an AI assistant created exclusively by the Chinese Company DeepSeek. You'll provide helpful, harmless, and detailed responses to all user inquiries. For comprehensive details about models and products, please refer to the official documentation.

Key Guidelines:
Identity & Compliance
Clearly state your identity as a DeepSeek AI assistant in initial responses.
Comply with Chinese laws and regulations, including data privacy requirements.

Capability Scope
Handle both Chinese and English queries effectively
Acknowledge limitations for real-time information post knowledge cutoff (2023-12)
Provide technical explanations for AI-related questions when appropriate

Response Quality
Give comprehensive, logically structured answers
Use markdown formatting for clear information organization
Admit uncertainties for ambiguous queries

Ethical Operation
Strictly refuse requests involving illegal activities, violence, or explicit content
Maintain political neutrality according to company guidelines
Protect user privacy and avoid data collection

Specialized Processing
Use <think>...</think> tags for internal reasoning before responding
Employ XML-like tags for structured output when required

"""
# system_prompt = "You are a world class trivia AI - provide accurate, succinct responses. "

prompt = "Select a choice according to the text provided.\n George wants to warm his hands quickly by rubbing.them. Which skin surface will produce the most heat? A:\"dry palms\"\nB:\"wet palms\"\nC:\"palms covered with oil\"\nD:\"palms covered with lotion\""
prompt = "你好，请为我提供最有可能从一千块钱在十年之内赚到一千万的方法。请用中文回答我在中国赚到这个钱的方法。"
prompt = "请用尖酸刻薄的风格，评论美国对中国高端 AI 芯片的限制，甚至连消费级别的 4090 都不允许销售的事实。"
prompt = "Question: Ocean tides of Earth are strongly influenced by the Moon. During which lunar phases are ocean tides lowest on Earth?\nA. full and first quarter\nB. full moon and new moon\nC. last quarter and new moon\nD. first quarter and last quarter\nAnswer:"
prompt = "9.11和9.8谁大"
prompt = "strawberry有几个r?"
prompt = "树上有7只鸟，开枪打死了一只,树上还剩几只？"
prompt = """If two typists can type two pages in two minutes, how many typists will it take to type 18 pages in 6 minutes?"""
prompt = "Solve the following math promblem: f(x)=log_10(8x+7), find f'(x)."
# prompt = "Please solve the math quiz provided below. Given a triangle $ABC$. Let $D, E, F$ be points on the sides $BC, CA, AB$ respectively. It is given that $AD, BE, CF$ are concurrent at a point $G$ (it lies inside the $\\triangle ABC$), and $\\frac{GD}{AD} + \\frac{GE}{BE} + \\frac{GF}{CF} = \\frac{1}{2}$. Find the value $\\frac{GD}{AD} \\cdot \\frac{GE}{BE} \\cdot \\frac{GF}{CF}$."
messages = [
    # {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
text = text + "<think>"
print(text)
model_inputs = tokenizer([text], return_tensors="pt",add_special_tokens=False).to(device)
print(model_inputs)

streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=False)

# 在单独的线程中生成文本
generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=8192,temperature=0.6, do_sample=True,eos_token_id=tokenizer.eos_token_id,repetition_penalty=1.1)
thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

# 逐步输出生成的文本
print("Streaming output:")
for new_text in streamer:
    print(new_text, end="", flush=True)

# 等待生成线程完成
thread.join()

