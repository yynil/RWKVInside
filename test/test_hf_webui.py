import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import threading

model_path = "/media/alic-li/WDdata011/ARWKV_7b/"
device = "cuda:0"

# 加载模型和分词器
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(model_path).to(dtype=torch.float16)
    model = model.to(device)
    model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

system_prompt = "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process are enclosed within <think> </think>, i.e., <think> reasoning process here </think>**Final Answer:**\nanswer here. "

def convert_history_to_messages(history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg is not None:
            messages.append({"role": "assistant", "content": bot_msg})
    return messages

def convert_messages_to_ds(messages):
    bos = "<｜begin▁of▁sentence｜>"
    user = "<｜User｜>"
    assistant = "<｜Assistant｜>"
    text = f"{bos}{system_prompt}"
    for index, conv in enumerate(messages):
        if index % 2 == 0:
            text += f"{user}{conv['content']}\n"
        else:
            text += f"{assistant}{conv['content']}\n"
    text += f"{assistant}\n<think>"
    return text

def stream_chat(prompt, history):
    messages = convert_history_to_messages(history)
    messages.append({"role": "user", "content": prompt})
    eos = "<｜end▁of▁sentence｜>"
    
    text = convert_messages_to_ds(messages)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=8192,
        do_sample=True,
        top_p=1.0,
        stop_strings=[eos],
        tokenizer=tokenizer,
        temperature=0.9,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    response = ""
    for new_text in streamer:
        response += new_text
        if "</think>" in response:
            think_part = response.split("</think>")[0] + "</think>"
            answer_part = response.split("</think>")[-1]
            formatted_response = f"<details><summary> Think 🤔</summary>{think_part}</details>" + "Answer 😋 \n\n" +answer_part
        else:
            formatted_response = response
        yield history + [(prompt, formatted_response)]

def user(user_message, history):
    return "", history + [[user_message, None]]

def bot(history):
    prompt = history[-1][0]
    history[-1][1] = ""
    for updated_history in stream_chat(prompt, history[:-1]):
        yield updated_history

with gr.Blocks() as demo:
    # 引入MathJax
    # demo.queue().launch(server_name="10.46.43.129", server_port=18653)
    gr.HTML(value="""
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """)
    
    chatbot = gr.Chatbot(label="Chat with LLM", height=750, elem_id="chatbox", render_markdown=True)
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear Chat")

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue().launch(server_name="10.46.43.129", server_port=18653)
