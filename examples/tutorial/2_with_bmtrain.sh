python3 -m torch.distributed.launch --master_addr localhost --master_port 34123 --nproc_per_node $1 --nnodes 1 --node_rank 0 2_with_bmtrain.py
