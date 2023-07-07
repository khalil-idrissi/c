import logging
import transformers
import sys
logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)
import os
#import deepspeed
import sys
sys.stderr.write("*************************************************************************")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import tensor_parallel as tp
import time
import json
import uuid
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import termcolor
import torch
import transformers
from flask import jsonify
import kserve
from typing import Dict
#from blocks import MPTBlock
#from .attention import ATTN_CLASS_REGISTRY

from dotenv import load_dotenv
import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from torch import nn

load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

API_KEY = "your_api_key_here"
global world_size
world_size = int(os.getenv('WORLD_SIZE', '2'))
#deepspeed.init_distributed(dist_backend="mpi")

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
#from .attention import ATTN_CLASS_REGISTRY

import tensor_parallel as tp


def disable_torch_init():
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    transformers.modeling_utils._init_weights = False

def get_replit(model, trust_remote_code=True):
    #config = AutoConfig.from_pretrained("replit/replit-code-v1-3b", trust_remote_code=True)
    #config.attn_config['attn_impl'] = 'triton'
    model = AutoModelForCausalLM.from_pretrained("replit/replit-code-v1-3b", use_auth_token=HF_ACCESS_TOKEN, trust_remote_code=trust_remote_code)
    #model = model.to("cuda", dtype=torch.float16)
    model = tp.tensor_parallel(model, ["cuda:0", "cuda:1"])
    model.eval()
    model.seqlen = 2048
    print("model dtype:")
    print(model.dtype)
    #config = {
    #    "kernel_inject": True,
    #   "tensor_parallel": {"tp_size": 2},
    #   "dtype": "fp16",
    #   "enable_cuda_graph" : False
    #    }
    #world_size = int(os.getenv('WORLD_SIZE', '2'))
    #model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float, replace_with_kernel_inject=True)

    return model

def generate_text(tokenizer, model, prompt, max_tokens, n, temperature, stop):
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}
    generated_texts = []
    for _ in range(n):
        generated = model.generate(batch["input_ids"].to("cuda"), do_sample=True, min_new_tokens=max_tokens, max_new_tokens=max_tokens, temperature=temperature, eos_token_id=tokenizer.encode(stop)[0] if stop else None)
        generated_texts.append(tokenizer.decode(generated[0]))
    return generated_texts


class KServeModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        self.ready = True

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        api_key = headers.get("authorization")
        print(api_key)
        if api_key != f"Bearer {API_KEY}":
            return {"error": "Invalid API key"}

        instances = request.get("instances", [])
        if not instances or not all(isinstance(instance, dict) for instance in instances):
            return jsonify({"error": "Invalid input parameters"})
        st = time.time()
        generated_texts = []
        for instance in instances:
            prompt = instance.get("prompt", "")
            max_tokens = instance.get("max_tokens", 1024)
            n = instance.get("n", 1)
            temperature = instance.get("temperature", 0.7)
            stop = instance.get("stop", None)

            if not isinstance(prompt, str) or not isinstance(max_tokens, int) or not isinstance(n, int) or not isinstance(temperature, (int, float)) or (stop is not None and not isinstance(stop, str)):
                return jsonify({"error": "Invalid input parameters"}), 400
            generated_texts.extend(generate_text(tokenizer, model, prompt, max_tokens, n, temperature, stop))
        et = time.time()
        print(f"inference time is : {et-st}")
        return {"inference time" : et-st ,"predictions": [{"text": text} for text in generated_texts]}


def main():
    global model, tokenizer
    #local_rank = int(os.getenv('LOCAL_RANK', '0'))
    #world_size = int(os.getenv('WORLD_SIZE', '2'))
    #parser = ArgumentParser()
    #parser.add_argument("model", type=str, help="model to load, such as replit-code-v1-3b")
    model_path = r"/mnt/pvc/replit-code-v1"

    print("-- Loading original model weights from PVC (this will take a few seconds)")
    start = time.time()
    load = os.path.join("/mnt/pvc/replit-code-v1/", "model.bin")
    #config = {
    #    "kernel_inject": False,
    #    "tensor_parallel": {"tp_size": 4},
    #    "dtype": "fp16",
    #    "enable_cuda_graph" : False,
    #    "injection_policy" : {MPTBlock: ('attn.out_proj', 'ffn.down_proj')}
    #    }
    disable_torch_init()
    num_of_gpus = torch.cuda.device_count()
    print(num_of_gpus)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    print("-- Loading model weights from PVC (this will take a few seconds)")
    os.chdir(r"/app/replit-code-v1-3b")
    model = get_replit("replit/replit-code-v1-3b")
    #model = deepspeed.init_inference(model=model, config=config)
    #model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float16, injection_policy={MPTBlock: ('attn.out_proj.weight', 'ffn.down_proj.weight')})
    #model = deepspeed.init_inference(model, mp_size=world_size, replace_with_kernel_inject=False, replace_method="auto")
    print("-- Loading tokenize model")
    tokenizer = AutoTokenizer.from_pretrained("replit/replit-code-v1-3b", torch_dtype=torch.float16, use_auth_token=HF_ACCESS_TOKEN, trust_remote_code=True)
    #model = deepspeed.init_inference(model=model, dtype=torch.float16, replace_with_kernel_inject=True)
    end = time.time()
    print("----------------------------------------------------")
    print(f"Time for loading the model: {end-start}")
    kserve_model = KServeModel("replit-code-v1")
    s = time.time()
    print("-- Ready to serve inference requests :)")
    kserve.ModelServer().start([kserve_model])
    e = time.time()
    print(f"time for inference: {e-s}")

if __name__ == "__main__":
    main()
