Procedue:
1. Check if data is up to date. If there is new data, you need to clean them and update the Data folder.
2. Make sure to check and update tokenizer_config
3. Run the container and start training:

```
docker run --gpus all -it --rm model-serving python train.py --model_id Qwen/Qwen2.5-0.5B --max_steps 100 --wandb_key “Your wandb key" --hf_key "Your hf key" --max_seq_length 1024
```

4. You can do a quick sannity check to see if the model is properly trained:


5. Finally, test if model would run in a TGI container: 

```
docker run --gpus all -p 8080:80 -v "local/path/to/repo:/mnt" ghcr.io/huggingface/text-generation-inference:latest --model-id Qwen/Qwen2.5-0.5B --lora-adapters myadapter=/mnt/checkpoints/checkpoint-1000

curl localhost:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{
  "inputs": "###Human:I am depressed###Assistant:",
  "parameters": {
    "max_new_tokens": 100,
    "adapter_id": "myadapter"
  }
}'
```