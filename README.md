Procedue:
1. check if data is up to date. If there is new data, you need to clean them and update the Data folder.
2. make sure to check and update tokenizer_config
3. run the container and start training
4. use TGI for testing: 
```
model=teknium/OpenHermes-2.5-Mistral-7B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all --shm-size 1g -p 8080:80 -v $volume:/data \
    ghcr.io/huggingface/text-generation-inference:3.2.1 \
    --model-id $model
```