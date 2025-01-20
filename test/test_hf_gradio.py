import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
import threading
import sys
model_path = sys.argv[1]
device = "cuda:2"
# 加载模型和分词器
with torch.no_grad():
    model = AutoModelForCausalLM.from_pretrained(model_path).to(dtype=torch.float16)
    model = model.to(device)
    model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 将 Gradio 的 history 格式转换为 apply_chat_template 所需的格式
def convert_history_to_messages(history):
    messages = []
    for user_msg, bot_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if bot_msg is not None:
            messages.append({"role": "assistant", "content": bot_msg})
    return messages

def convert_messages_to_chatml(messages):
    text = ""
    for conv in messages:
        text += f"<|im_start|>{conv['role']}\n{conv['content']}<|im_end|>"
    text += "<|im_start|>assistant\n"
    return text

# 流式生成函数
def stream_chat(prompt, history):
    # 将历史对话转换为 apply_chat_template 所需的格式
    messages = convert_history_to_messages(history)
    messages.append({"role": "user", "content": prompt})
    
    # 将输入转换为模型输入格式
    # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text = convert_messages_to_chatml(messages)
    print(text)
    print('----------------')
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 创建流式输出器
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 在单独的线程中生成文本
    generation_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=4096,
        do_sample=True,
    )
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # 逐步输出生成的文本
    response = ""
    for new_text in streamer:
        response += new_text
        yield history + [(prompt, response)]  # 实时返回完整的对话历史

# 创建 Gradio 聊天界面
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="Chat with LLM", height=750)
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear Chat")

    def user(user_message, history):
        return "", history + [[user_message, None]]  # 添加用户消息到历史

    def bot(history):
        prompt = history[-1][0]  # 获取最新的用户输入
        history[-1][1] = ""  # 初始化机器人的回复为空
        for updated_history in stream_chat(prompt, history[:-1]):
            yield updated_history  # 实时更新对话历史

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    clear.click(lambda: None, None, chatbot, queue=False)

# 启动 Gradio 应用
demo.queue().launch(server_name="0.0.0.0")