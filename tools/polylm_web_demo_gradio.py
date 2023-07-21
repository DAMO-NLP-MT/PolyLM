import argparse
import os
import warnings
import mdtex2html
import gradio as gr

import re
pattern = re.compile("[\n]+")

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

from transformers.generation.utils import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="/mnt/user/E-xiangpeng.wxp-201390/convert_from_megatron_to_transformers/ckpts/ct_from_22350_step_11904_ecomm_sft_step_600_bf16",
                    choices=["DAMO-NLP-MT/polylm-multialpaca-13b"], type=str)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

if ('int8' in args.model_name or 'int4' in args.model_name) and num_gpus > 1:
    raise ValueError("Quantized models do not support model parallel. Please run on a single GPU (e.g., --gpu 0).")

logger.setLevel("ERROR")
warnings.filterwarnings("ignore")

model_path = args.model_name
if not os.path.exists(args.model_name):
    model_path = snapshot_download(args.model_name)

config = AutoConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

if num_gpus > 1:
    print("Waiting for all devices to be ready, it may take a few minutes...")
    with init_empty_weights():
        raw_model = AutoModelForCausalLM.from_config(config)
    raw_model.tie_weights()
    model = load_checkpoint_and_dispatch(
        raw_model, model_path, device_map="auto", no_split_module_classes=["GPT2Block"]
    )
else:
    print("Loading model files, it may take a few minutes...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).cuda()

print(model.dtype)

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history):
    query = input
    query = query.strip()
    query = re.sub(pattern, "\n", query)

    chatbot.append((query, ""))
    prompt = ""
    for i, (old_query, response) in enumerate(history):
        prompt += f"{old_query}\n\n" + f"{response}\n"
    prompt += f"{query}\n\n"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.cuda(),
            attention_mask=inputs.attention_mask.cuda(),
            max_length=max_length,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.02,
            num_return_sequences=1,
            eos_token_id=2,
            early_stopping=True)
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    chatbot[-1] = (parse_text(query), parse_text(response))
    history = history + [(query, response)]
    print("==========================================================================")
    print(f"chatbot is {chatbot}")
    print(f"history is {history}")
    print("==========================================================================")
    return chatbot, history


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], []


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">欢迎使用 PolyLM 多语言人工智能助手！</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(
                0, 2048, value=1024, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01,
                              label="Top P", interactive=True)
            temperature = gr.Slider(
                0.01, 1, value=0.7, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])  # (message, bot_message)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history], [chatbot, history],
                    show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=False, inbrowser=True)
