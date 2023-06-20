import os
import re
import sys
import json
import pickle
import argparse
import threading
import numpy as np
from tqdm import tqdm

from typing import List, Union

import torch
import torch.nn.functional as F
from peft import PeftModel

from accelerate import init_empty_weights, infer_auto_device_map

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaModel, OPTForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def get_device_map(model_name, device, do_int8):
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_config(config)

    d = {device: "24GiB"}
    # for i in range(1, 4):
    #     d[i] = "24GiB"
    device_map = infer_auto_device_map(
        model, max_memory=d, dtype=torch.int8 if do_int8 else torch.float16,
        no_split_module_classes=["BloomBlock", "OPTDecoderLayer", "LlamaDecoderLayer"]
    )
    print(device_map)
    del model
    return device_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", '-n', help='name of the model, should be unique', required=True)
    parser.add_argument("--base_model", '-m', help='path of the model', required=True)
    parser.add_argument("--lora_weights", '-l', help='path of the lora weights', default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--do_int8", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--port", type=int, default=12333)
    parser.add_argument("--temperature", "-t", type=float, default=0.1)
    parser.add_argument("--top_p", "-p", type=float, default=0.75)
    parser.add_argument("--top_k", "-k", type=int, default=40)
    parser.add_argument("--num_beams", "-b", type=int, default=4)
    parser.add_argument("--max_new_tokens", "-s", type=int, default=128)
    parser.add_argument("--no_cuda", action="store_true")
    
    
    args = parser.parse_args()
    
    model_name = args.model_name
    base_model = args.base_model
    lora_weights = args.lora_weights
    device = args.device
    
    print('Import finished...')
    print(f'Model name: {model_name}')
    print(f'Model path: {base_model}')
    print(f'LoRA weights path: {lora_weights}')
    
    if base_model == "gpt2-xl":
        tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
        model = GPT2LMHeadModel.from_pretrained(
            args.base_model,
            device_map=get_device_map(args.base_model, args.device, args.do_int8) if not args.no_cuda else None,
            torch_dtype=torch.int8 if args.do_int8 else torch.float16 if not args.no_cuda else torch.float32,
            load_in_8bit=args.do_int8,
        )
    elif base_model == "llama-7b":
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        model = LlamaForCausalLM.from_pretrained(
            "huggyllama/llama-7b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=get_device_map("huggyllama/llama-7b", args.device, args.do_int8) if not args.no_cuda else None,
        )
    elif base_model == "llama-13b":
        tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-13b")
        model = LlamaForCausalLM.from_pretrained(
            "huggyllama/llama-13b",
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=get_device_map("huggyllama/llama-13b", args.device, args.do_int8) if not args.no_cuda else None,
        )
    elif base_model == "mpt-7b":
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        model = AutoModelForCausalLM.from_pretrained(
            "mosaicml/mpt-7b",
            trust_remote_code=True,
            device_map=get_device_map("mosaicml/mpt-7b", args.device, args.do_int8) if not args.no_cuda else None,
            torch_dtype=torch.float16,
        )
    elif base_model == "gal-6.7b":
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-6.7b")
        model = OPTForCausalLM.from_pretrained(
            "facebook/galactica-6.7b", 
            device_map=get_device_map("facebook/galactica-6.7b", args.device, args.do_int8) if not args.no_cuda else None, 
            torch_dtype=torch.float16,
        )
    elif base_model == "vicuna-7b":
        tokenizer = AutoTokenizer.from_pretrained("/home/daven/llm/FastChat/vicuna-7b")
        model = OPTForCausalLM.from_pretrained(
            "/home/daven/llm/FastChat/vicuna-7b", 
            device_map=get_device_map("/home/daven/llm/FastChat/vicuna-7b", args.device, args.do_int8) if not args.no_cuda else None, 
            torch_dtype=torch.float16,
        )
    elif base_model == "alpaca-7b":
        tokenizer = LlamaTokenizer.from_pretrained("chainyo/alpaca-lora-7b")
        model = LlamaForCausalLM.from_pretrained(
            "chainyo/alpaca-lora-7b",
            load_in_8bit=True,
            device_map=get_device_map("chainyo/alpaca-lora-7b", args.device, args.do_int8) if not args.no_cuda else None, 
            torch_dtype=torch.float16,
        )
    else:
        """
        K2 (GeoLLaMA) series Models
        """
        if lora_weights is None:
            load_8bit = True
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                device_map=get_device_map(args.base_model, args.device, args.do_int8) if not args.no_cuda else None,
                torch_dtype=torch.float16
            )
            model.config.pad_token_id = tokenizer.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2
            
            if not load_8bit:
                model.half()
                print("We need to make the model smaller")
        else:
            load_8bit = False
            tokenizer = LlamaTokenizer.from_pretrained(base_model)
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                device_map=get_device_map(args.base_model, args.device, args.do_int8) if not args.no_cuda else None,
                torch_dtype=torch.float16
            )
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map=get_device_map(args.base_model, args.device, args.do_int8) if not args.no_cuda else None,
            )
            model.config.pad_token_id = tokenizer.pad_token_id = 0
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

            if not load_8bit:
                model.half()
            model.eval()
            if torch.__version__ >= "2" and sys.platform != "win32":
                model = torch.compile(model)
    # tokenizer.pad_token_id = -1
    print('Model loaded.')
    
    prompter = Prompter("")
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    
    # if not args.do_int8:
    #     model.half()  # seems to fix bugs for some users.

    model.eval()
    # if torch.__version__ >= "2" and sys.platform != "win32":
    #     model = torch.compile(model)
        
        
    with open('input_ls.json', 'r') as file:
        question_ls = json.load(file)
        
    all_result = []
    for item in tqdm(question_ls):
        prompt = prompter.generate_prompt(item, "")
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            no_repeat_ngram_size=2,
            # forced_eos_token_id=[tokenizer('.')["input_ids"][-1]],
            # early_stopping=True
        )
        
        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=args.max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        
        all_result.append(
            {
                'question': item,
                'answer': prompter.get_response(output)
            }
        )
    json.dump(all_result, open(f"qa_result_{args.model_name}", "w"))
    