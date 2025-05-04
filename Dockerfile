# Pull the latest huggingface base image
FROM huggingface/transformers-pytorch-gpu
# let up work space (not necessary but good practice)
WORKDIR /workspace
COPY . ./
# install requirements
RUN pip install --no-cache-dir -r requirements.txt