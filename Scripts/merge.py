import json
import torch
import os
import copy
import bitsandbytes as bnb
from bitsandbytes.functional import dequantize_4bit
from transformers import (
    AutoModelForCausalLM,
)
from peft import PeftModel
from peft.utils import _get_submodules
from tokenizer_config import load_tokenizer

def save_model(model, tokenizer, to):
    print(f"Saving dequantized model to {to}...")
    model.save_pretrained(to, safe_serialization=True)
    tokenizer.save_pretrained(to)
    config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
    config_data.pop("quantization_config", None)
    config_data.pop("pretraining_tp", None)
    with open(os.path.join(to, 'config.json'), 'w') as config:
        config.write(json.dumps(config_data, indent=2))

def dequantize_model(model, to='./dequantized_model', dtype=torch.float16, device="cuda"):
    """
    'model': the peftmodel you loaded with qlora.
    'tokenizer': the model's corresponding hf's tokenizer.
    'to': directory to save the dequantized model
    'dtype': dtype that the model was trained using
    'device': device to load the model to
    """

    os.makedirs(to, exist_ok=True)

    cls = bnb.nn.Linear4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, cls):
                print(f"Dequantizing `{name}`...")
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state.dtype = dtype

                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(module.in_features, module.out_features, bias=None, dtype=dtype)
                new_module.weight = torch.nn.Parameter(weights)
                new_module.to(device=device, dtype=dtype)

                parent, target, target_name = _get_submodules(model, name)
                setattr(parent, target_name, new_module)

        # a hack, setting this to avoid hf's saving error because hf
        # itself does not support saving a model that is registered to be loaded in 4bit.
        model.is_loaded_in_4bit = False

        print("Saving dequantized model...")
        model.save_pretrained(to)
        #tokenizer.save_pretrained(to)
        config_data = json.loads(open(os.path.join(to, 'config.json'), 'r').read())
        config_data.pop("quantization_config", None)
        config_data.pop("pretraining_tp", None)
        with open(os.path.join(to, 'config.json'), 'w') as config:
            config.write(json.dumps(config_data, indent=2))

        return model

def merge(adapter, args, quantization_config):

    model_name = args.model_id
    tokenizer = load_tokenizer(args)
    compute_dtype = torch.float16

    try:
        print(f"Starting to load the model {model_name} into memory")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            token = args.hf_key
        )
        model = dequantize_model(model, to='./dqz_model/',dtype=compute_dtype)
        print(model)
        model = PeftModel.from_pretrained(model, adapter)
        print(model)
        model = model.merge_and_unload()
        print(model)

        print(f"Successfully loaded the model {model_name} into memory")
        save_model(model, tokenizer, "./model/") # changed
        print(f"Merged model saved")
    except Exception as e:
        print(f"An error occurred: {e}")
    