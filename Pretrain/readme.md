# EnzymeFlow Hierarchical Pre-training

![pretrain](../image/pretrain.jpg)

Pre-training on PDBBind2020

1. Download PDBBind2020 data from [link](https://www.pdbbind-plus.org.cn/download), then put pocket data under ./data/pdb folder.

2. Run process_data.py to process rawdata to meatadata.

   (a) read pocket rawdata (pocket pdb and ligand mol2 files) from ./data/pdb.
   
   (b) pocket metadata will be saved into ./data/processed.
   
   (c) metadata.csv will be saved into ./data along with a label file kdvalue.csv.
   
   (d) a toy example is provided.

3. configs.py contain all pre-training configurations and hyperparameters.

4. Train model using train_ddp.py for parallal training with multi-gpus (we trained with 4gpus).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py
```
