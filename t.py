from typing import Any
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import deepspeed
class Model:
    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoConfig

        config = AutoConfig.from_pretrained("replit/replit-code-v1-3b",trust_remote_code=True)
        #config.attn_config['attn_impl'] = 'triton'
        config.no_bias = True
        config.verbose = 0
        config.norm_type = "layernorm"
        self._tokenizer = AutoTokenizer.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained('replit/replit-code-v1-3b', trust_remote_code=True, config=config)
        #self._model = deepspeed.init_inference(model=self._model, tensor_parallel={"tp_size":2}, dtype=torch.float16)
        self._model.to("cuda")
        #print(self._model.module)
        #print(dir(self._model))
        #print(AutoTP().get_module_list(self._model))
        #print(get_module_list(self._model))

        #print(self._model.children)

    def predict(self, model_input: Any, max_length=100, do_sample=True, top_p=0.95, top_k=4, temperature=0.2, num_return_sequences=1) -> str:
        x = self._tokenizer.encode(model_input, return_tensors='pt')
        y = self._model.generate(x.cuda(), max_length=max_length, do_sample=do_sample, top_p=top_p, top_k=top_k, temperature=temperature, num_return_sequences=num_return_sequences, eos_token_id=self._tokenizer.eos_token_id)
        generated_code = self._tokenizer.decode(y[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return generated_code

model = Model()
s = time.time()
model.load()
e = time.time()
print(f"time for loading the model is : {e-s}")
start = time.time()
print(model.predict("def fibonacci(n):"))
end = time.time()
print(f"time for inference is {end-start}")
