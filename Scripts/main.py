import argparse
from train import training_function

def parse_arge():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    # add model id and dataset path argument
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model id to use for training.",
    )
    parser.add_argument("--resume_from_checkpoint", type=bool, default=False, help="Continue training this checkpoint")
    parser.add_argument("--train_dataset_path", type=str, default="Data/all.json", help="Path to train dataset.")
    parser.add_argument("--max_steps", type=int, default=100, help="Number of steps to train for.")
    parser.add_argument("--wandb_key", type=int, required=True, help="Your wandb API key")
    parser.add_argument("--hf_key", type=int, required=True, help="Your huggingface API key")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size to use for training.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate to use for training.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to use for training.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="batches accumulated before backprop.")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Sequence length for training")
    parser.add_argument(
        "--merge_weights",
        type=bool,
        default=True,
        help="Whether to merge LoRA weights with base model.",
    )
    parser.add_argument(
        "--load_in_4bit",
        type=bool,
        default=True,
        help="Whether train with LoRA or QLoRA",
    )
    parser.add_argument(
        "--adapter_checkpoints",
        type=bool,
        default=False,
        help="Whether should save the LoRA weights",
    )
    args = parser.parse_known_args()
    return args

def main():
    args, _ = parse_arge()
    training_function(args)

# run the function
if __name__ == "__main__":
    main()