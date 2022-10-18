# RUN


```
# 8GPU
python -m torch.distributed.launch --nproc_per_node=8 DDP_train_1b.py
```


# 8GPU+
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0  --master_addr="33.255.83.232" --master_port=34567   DDP_train_1b.py

python -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=1  --master_addr="33.255.83.232" --master_port=34567   DDP_train_1b.py
```

