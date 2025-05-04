
# Each model architecture might come with different pad tokens
# Make sure you configure this file before training

# For example Qwen tokenizer has the following configuration
# def load_tokenizer(args):
#     tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
#     tokenizer.pad_token = "<|im_end|>"
#     tokenizer.pad_token_id = 151645
#     tokenizer.padding_side = 'left'


from transformers import AutoTokenizer

def load_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, token = args.hf_key)
    tokenizer.pad_token = "<|im_end|>"
    tokenizer.pad_token_id = 151645
    tokenizer.padding_side = 'left'
    return tokenizer