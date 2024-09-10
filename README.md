# EnzymeFlow: Generating Reaction-specific Enzyme Catalytic Pockets through Flow-Matching and Co-Evolutionary Dynamics

## Model
![enzymeflow](./image/enzymeflow.jpg)

### Requirement
```
torch==2.4.1 (>=2.0.0)
numpy==1.24.4
pytorch-warmup==0.1.1
POT==0.9.4
rdkit==2023.9.5
biopython==1.84
torch_geometric==2.4.0
tmtools==0.2.0
scipy==1.14.0
geomstats==2.7.0
tqdm==4.65.0
```

### Model Training

1. ```configs.py``` contain all training configurations and hyperparameters.

2. Train model using ```train_ddp.py``` for parallal training with multi-gpus (we trained with 4 A40 gpus).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py
```

3. The training loads [pre-trained model](https://github.com/WillHua127/EnzymeFlow/tree/main/Pretrain). You may also train from scratch by setting the configs in ```configs.py```, setting parameters ```ckpt_from_pretrain=False pretrain_ckpt_path=None```.



### Model Inference


## Data
![pocket](./image/pocket.jpg)

### 1. Enzyme Pocket and Substrate, Product Molecule Rawdata
   
   $~~~~$ (a) [molecule_structures](https://github.com/WillHua127/EnzymeFlow/tree/main/data/molecule_structures) folder in ```./data``` contain all substrate and product molecules, can be downloaded at link.
   
   $~~~~$ (b) [pocket_fixed_residues/pdb_10A](https://github.com/WillHua127/EnzymeFlow/tree/main/data/pocket_fixed_residues/pdb_10A) folder in ```./data``` contain all enzyme pockets, can be downloaded at link.

   $~~~~$ (c) We provide [rawdata-40%homology](https://github.com/WillHua127/EnzymeFlow/blob/main/data/rawdata_cutoff-0.4.csv) and [metadata-40%homology](https://github.com/WillHua127/EnzymeFlow/blob/main/data/metadata_cutoff-0.4.csv) with 40% homologys in ```./data``` folder. More rawdata can be downloaded at link.



### 2. Co-evolution and MSA
   
   $~~~~$ (a) [rxn_to_smiles_msa.pkl](https://github.com/WillHua127/EnzymeFlow/blob/main/data/rxn_to_smiles_msa.pkl) in ```./data``` contain reaction MSAs.
   
   $~~~~$ (b) [uid_to_protein_msa.pkl](link) in ```./data``` contain enzyme MSAs, can be downloaded at link.

   $~~~~$ (c) [vocab.txt](https://github.com/WillHua127/EnzymeFlow/blob/main/data/vocab.txt) in ```./data``` is co-evolution vocabulary.


### When the raw enzyme pockets, molecules, co-evolution data are ready, then

### 3. Process rawdata into metadata by running ```process_data.py```.
   
   $~~~~$ (a) Remember to change the configs ```--rawdata_file_name```, e.g., ```python process_data.py --rawdata_file_name rawdata_cutoff-0.4.csv```. Warning: we have absolute path in ```metadata.csv```, so you might need to change it to your path.

   
### 4. Processed Metadata.

   $~~~~$ (a) Processed data will be saved into ```./data/processed``` folder, with:
   
   $~~~~$ (b) processed enzyme in ```./data/processed/protein``` folder.

   $~~~~$ (c) processed substrate in ```./data/processed/ligand``` folder.

   $~~~~$ (d) processed co-evolution in ```./data/processed/msa``` folder.

   $~~~~$ (e) processed produuct in ```./data/processed/product``` folder.

   $~~~~$ (f) a toy [example](https://github.com/WillHua127/EnzymeFlow/tree/main/data/processed) is provided.


### 5. Evaluation Sample.
   
   $~~~~$ (a) We provide [eval-rawdata](https://github.com/WillHua127/EnzymeFlow/blob/main/data/eval-data_cutoff-0.1_unique-subs-enz_100.csv) and [eval-metadata](https://github.com/WillHua127/EnzymeFlow/blob/main/data/metadata_eval.csv) in ```./data``` folder. Warning: we have absolute path in metadata.csv, so you might need to change it to your path.

   $~~~~$ (b) We provide [unprocessed-eval-data](https://github.com/WillHua127/EnzymeFlow/tree/main/data/raw_eval_data) in ```./data/raw_eval_data``` folder.

   $~~~~$ (c) We provide [processed-eval-data](https://github.com/WillHua127/EnzymeFlow/tree/main/data/processed_eval) in ```./data/processed_eval``` folder.

   $~~~~$ (d) You can also process evaluation data by running ```process_data.py```. Remeber to change the configs, e.g., ```python process_data.py --rawdata_file_name eval-data_cutoff-0.1_unique-subs-enz_100.csv --metadata_file_name metadata_eval.csv```.






## Further Statistics
![distribution](./image/distribution.jpg)

