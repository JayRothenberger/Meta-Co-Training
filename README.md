# Meta Co Training

requirements: timm, pytorch, torchvision, wandb, numpy, 

To reproduce:
0. install the requisite dependencies
1. embed the dataset of your choice using the models available (DINOv2, CLIP, SwAV, EsViT, MAE)
2. launch MCT_experiment.py using torchrun as shown below
3. view the results on Weights and Biases, or load the model checkpoints saved in the grid search path under ./checkpoints

The embed_dataset file expects the imagenet dataset to be in ../imagenet.  If it is not, you can change lines 1625 and 1626 of the file to include the correct filepath.  We note that some experiments are more easily reproduced in terms of the amount of time they take to perform.  If you would instead like to generate embeddings for other datasets, the lines corresponding to those datasets between 1627 and 1653.

wandb login <your wandb key>

example usage: embed_dataset.py

python embed_dataset.py --embedding DINOv2 # generate DINOv2 embedding

example usage: MCT_experiment.py

wandb login <your wandb key>

srun torchrun \
--nnodes 1 \ # just one machine
--nproc_per_node 1 \ # single gpu
--rdzv_id $RANDOM \
--rdzv_backend c10d \
--rdzv_endpoint "127.0.0.1:64425" \ # local machine
MCT_bench.py