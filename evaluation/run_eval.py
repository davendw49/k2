# Code adapted from: https://github.com/zsc/llama_infer
# All copyrights belong to the original code owner.
import json

import torch
import argparse
import threading
import sys
from accelerate import init_empty_weights, infer_auto_device_map
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaModel, OPTForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel

from peft import PeftModel
from typing import List, Union
from tqdm import tqdm
import re
import numpy as np
import pickle
import torch.nn.functional as F
import os


def get_device_map(model_name, device, do_int8):

    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

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

def run_generate(input):
    # prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{input}\n\n### Response:\n{x1}"
    generate_kwargs = {
        "max_new_tokens": 700,
        "min_new_tokens": 10,
        "temperature": 0.2,
        "do_sample": True,  # The three options below used together leads to contrastive search
        "top_p": 0.75,
        "top_k": 40,
        "num_beams": 4,
        "penalty_alpha": 1.1,
    }
    with torch.no_grad():
        input_ids = tokenizer(input, return_tensors="pt", padding=True).input_ids
        assert len(input_ids) == 1, len(input_ids)
        if input_ids[0][-1] == 2:  # 2 is EOS, hack to remove. If the prompt is ending with EOS, often the generation will stop abruptly.
            input_ids = input_ids[:, :-1]
        input_ids = input_ids.to("cpu")

        generated_ids = model.generate(
            input_ids,
            **generate_kwargs
        )

        result = tokenizer.batch_decode(generated_ids.cpu(), skip_special_tokens=True)
        return result[0].strip()[len(input):].strip()

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", '-n', help='name of the model, should be unique', required=True)
    parser.add_argument("--base_model", '-p', help='path of the model', required=True)
    parser.add_argument("--lora_weights", '-l', help='path of the lora weights', default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--do_int8", action="store_true")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    parser.add_argument("--port", type=int, default=12333)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--geobenchmark", type=str, default="npee")
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
        tokenizer = LlamaTokenizer.from_pretrained("/home/daven/llm/qokori/llama-7b")
        model = LlamaForCausalLM.from_pretrained(
            "/home/daven/llm/qokori/llama-7b",
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
    
    with open(f'../data/geobench/{args.geobenchmark}.json', 'r') as f:
        source_target = json.load(f)

    
    all_softmax = {}
    for question_type in ['choice', 'tf']:
        if question_type == 'choice' and args.geobenchmark == 're':
            continue
        print(f"Task: {'True-False' if question_type == 'tf' else 'Multiple Choice'}")
        question_data = {}
        for the_answer_is in ['wa', 'woa']:
            print(f"Generate the answer with{'' if the_answer_is == 'wa' else 'out'}")
            softmax = []
            source = source_target['source'][question_type][the_answer_is]
            for i in tqdm(range(len(source))):
                input = source[i]
                input_ids = tokenizer(input, return_tensors='pt')
                outputs = model(input_ids["input_ids"])
                softmax_res = F.softmax(outputs["logits"][0][-1].float()).detach()
                softmax.append(softmax_res)
            question_data[the_answer_is] = softmax
        all_softmax[question_type] = question_data
            
    with open(f'./pickle/softmax_{model_name}_{args.geobenchmark}.pickle', 'wb') as f:
        pickle.dump(all_softmax, f)
        
    os.system(f'python ./post_process.py -m {model_name} -b {args.geobenchmark}')
    
"""
Here is an example:

- For k2 series model with lora
> python run_eval.py --model_name k2_ni --base_model /home/daven/llm/qokori/llama-2023-05-07-15-10/checkpoint-19180/ --lora_weights /home/daven/llm/qokori/qokori-sft/outputs/geo_llama_ni/

- For k2 series model without lora
> python run_eval.py --model_name geollama_19180 --base_model /home/daven/llm/qokori/llama-2023-05-07-15-10/checkpoint-19180/

- For non-k2 series model
> python run_eval.py --model_name gpt2_xl --base_model gpt2-xl
"""