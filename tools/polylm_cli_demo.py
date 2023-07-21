import argparse
import os
import platform
import warnings

import re
pattern = re.compile("[\n]+")

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download

from transformers.generation.utils import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="DAMO-NLP-MT/polylm-multialpaca-13b", 
                    choices=["DAMO-NLP-MT/polylm-multialpaca-13b"], type=str)
parser.add_argument("--multi_round", action="store_true",
                    help="Turn multiple rounds interaction on.")
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
num_gpus = len(args.gpu.split(","))

if args.model_name in ["DAMO-NLP-MT/polylm-multialpaca-13b-int8", "DAMO-NLP-MT/polylm-multialpaca-13b-int4"] and num_gpus > 1:
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
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto").cuda()

def clear():
    os.system('cls' if platform.system() == 'Windows' else 'clear')
    
def main():
    print("欢迎使用 PolyLM 多语言人工智能助手！输入内容即可进行对话。输入 clear 以清空对话历史，输入 stop 以终止对话。")
    prompt = ""
    while True:
        query = input()
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            if args.multi_round:
                prompt = ""
            clear()
            continue
        
        text = query.strip()
        text = re.sub(pattern, "\n", text)
        if args.multi_round:
            prompt += f"{text}\n\n"
        else:
            prompt = f"{text}\n\n"
        
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.cuda(), 
                attention_mask=inputs.attention_mask.cuda(), 
                max_length=1024, 
                do_sample=True, 
                top_p=0.8, 
                temperature=0.7,
                repetition_penalty=1.02,
                num_return_sequences=1, 
                eos_token_id=2,
                early_stopping=True)
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            if args.multi_round:
                prompt += f"{response}\n"
            print(f">>> {response}")
    
if __name__ == "__main__":
    main()
