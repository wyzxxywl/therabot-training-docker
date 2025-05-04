import torch
import wandb
import os
import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, prepare_model_for_kbit_training
from merge import merge
from tokenizer_config import load_tokenizer

def load_model_and_config(args):
    #use bf16 and FlashAttention if supported
    if torch.cuda.is_bf16_supported() and args.flash_attn:
        os.system('pip install flash-attn==2.6.3')
        compute_dtype = torch.bfloat16
        attn_implementation = 'flash_attention_2'
    else:
        compute_dtype = torch.float16
        attn_implementation = 'sdpa'

    # Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
        
    # Load model depends on precision
    if args.load_in_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            token = args.hf_key
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            device_map="auto",
            attn_implementation=attn_implementation,
            token = args.hf_key
        )
    
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={'use_reentrant':True})
    
    # Load tokenizer
    tokenizer = load_tokenizer(args)

    # lora config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
    )

    # set where and how to sage adapter endpoints
    if args.adapter_checkpoints:
        output_dir = f"./Output/{args.model_id}/checkpoints"
        save_strategy = "steps"
    else:
        output_dir = f"./Output/{args.model_id}/final_adapter"
        save_strategy = "no"

    return model, tokenizer, peft_config, output_dir, save_strategy, bnb_config

def begin_training(args, output_dir, save_strategy, model, tokenizer, peft_config, train_dataset):
    # Define training args, change these hyperparameters as needed
    training_args = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        optim="paged_adamw_8bit",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        learning_rate=args.lr,
        max_steps=args.max_steps,
        logging_dir=f"{output_dir}/logs/",
        logging_strategy="steps",
        save_strategy = save_strategy,
        save_steps = 10,
        save_total_limit = 10,
        logging_steps=10,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        dataset_text_field="text",
        max_seq_length = args.max_seq_length,
    )

    # Create Trainer instance
    trainer = SFTTrainer(
        model=model,
        processing_class = tokenizer,
        peft_config=peft_config,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    if args.resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint(output_dir)
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
    
    # free up spaces
    del model
    del trainer
    torch.cuda.empty_cache()

def merge_and_save(args, output_dir, bnb_config):
    # final adapter
    final_adapter = get_last_checkpoint(output_dir) if args.adapter_checkpoints else output_dir
    
    # merge
    if args.merge_weights and not args.load_in_4bit:
        # use the default merge_and_unload method if model not load in 4bit
        # load PEFT model in fp16
        model = AutoModelForCausalLM.from_pretrained(
            final_adapter,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            token = args.hf_key
        )  
        # Merge LoRA and base model and save
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained("./model/",safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(final_adapter, trust_remote_code=True)
        tokenizer.save_pretrained("./model/")
    elif args.merge_weights and args.load_in_4bit:
        merge(final_adapter, args, bnb_config)
    else:
        return

def training_function(args):
    # set seed
    set_seed(args.seed)
    # report to wandb
    wandb.login(key=args.wandb_key)
    # load train Datasets
    train_dataset = load_dataset("json", data_files=args.train_dataset_path)["train"]
    # load model and config
    model, tokenizer, peft_config, output_dir, save_strategy, bnb_config = load_model_and_config(args)
    # start training
    begin_training(args, output_dir, save_strategy, model, tokenizer, peft_config, train_dataset)
    # merge and save the model
    merge_and_save(args, output_dir, bnb_config)