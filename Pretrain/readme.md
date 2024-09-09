# EnzymeFlow Hierarchical Pre-training

Pre-training on pdbbind2020


1. run process_data.py to process rawdata to meatadata.
   -- read pocket rawdata (pocket pdb and ligand mol2 files) from ./data/pdb
   -- pocket metadata will be saved into ./data/processed
   -- metadata.csv will be saved into ./data along with a kd-value label file kdvalue.csv
   -- toy example is provided
