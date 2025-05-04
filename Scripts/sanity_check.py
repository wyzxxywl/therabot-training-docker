import argparse
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument("--hf_key", type=str, required=True, help="Your huggingface API key")
    parser.add_argument("--base_model", type=str, help="base model used to training")
    parser.add_argument("--adapter", type=str, help="trained adapter after training")
    parser.add_argument("--evaluate_adapter", type=str, help="whether to evaluate adapter + base or merged model")
    parser.add_argument("--flash_attn", type=bool, default=False, help="To use flash attention or not")
    parser.add_argument(
        "--model_id",
        type=str,
        help="Model id to use for training.",
    )
    args = parser.parse_known_args()
    return args

# load the most recent model

def load_model(args):
    print("Evaluating merged model")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    return model, tokenizer

def load_model_adapter(args):
    print("Evaluating adapter + base model")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config, device_map='cuda', token=args.hf_key)
    model = PeftModel.from_pretrained(model, args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)

    return model, tokenizer

def evaluate(model, tokenizer):
    prompt = ""
    while True:
        msg = input("You: ")
        if msg.lower() == 'quit':
            break
        prompt += f"###Human:{msg}###Assistant:"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out_ids = model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=False,    
                temperature=1.0,    
                top_k=50,          
                top_p=0.9,          
                repetition_penalty=1.0 
            )
            response = tokenizer.decode(out_ids[0], skip_special_tokens=False)
        print(f"Assistant: {response}")
        prompt += response



def main():
    args, _ = parse_arge()
    if args.evaluate_adapter == "True":
        model, tokenizer = load_model_adapter(args)
    else:
        model, tokenizer = load_model(args)
    evaluate(model, tokenizer)

# run the function
if __name__ == "__main__":
    main()