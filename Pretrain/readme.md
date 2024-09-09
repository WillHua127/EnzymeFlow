# EnzymeFlow Hierarchical Pre-training

Pre-training on PDBBind2020

1. Download PDBBind2020 data from [link](https://www.pdbbind-plus.org.cn/download), then put pocket data under ./data/pdb folder.

2. run process_data.py to process rawdata to meatadata.

   (a) read pocket rawdata (pocket pdb and ligand mol2 files) from ./data/pdb
   
   (b) pocket metadata will be saved into ./data/processed
   
   (c) metadata.csv will be saved into ./data along with a kd-value label file kdvalue.csv
   
   (d) toy example is provided
