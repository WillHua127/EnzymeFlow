# EnzymeFlow: Generating Reaction-specific Enzyme Catalytic Pockets through Flow-Matching and Co-Evolutionary Dynamics

Fine-tuning or Training on Catalytic Pockets. 
Pre-training can be found at [link](https://github.com/WillHua127/EnzymeFlow/tree/main/Pretrain). 

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

## Model Training

1. Please refer to the below, to see how we prepare training data.

2. ```configs.py``` contain all training configurations and hyperparameters.

3. Train model using ```train_ddp.py``` for parallal training with multi-gpus (we trained with 4 A40 gpus).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_ddp.py
```

4. The training loads [pre-trained model](https://github.com/WillHua127/EnzymeFlow/tree/main/Pretrain). You may also train from scratch by setting the configs in ```configs.py```, setting parameters ```ckpt_from_pretrain=False pretrain_ckpt_path=None```.



## Model Inference


## Baseline Experiments

### 1. RFDiff-AA
   
For RFDiffAA and LigandMPNN, please refer to [RFDiffAA-official](https://github.com/baker-laboratory/rf_diffusion_all_atom) and [LigandMPNN-official](https://github.com/dauparas/LigandMPNN?tab=readme-ov-file). For each enzyme-reaction pair in evaluation data, we use [RFDiffAA](https://github.com/baker-laboratory/rf_diffusion_all_atom) with default params to generate 100 catalytic pockets (with 32 residues) for each unique substrate. Then we use [LigandMPNN](https://github.com/dauparas/LigandMPNN?tab=readme-ov-file) to perform sequence prediction (inverse folding) on the generated catalytic pockets post-hoc.

### 2. Enzyme Commission Classifcation

Baselines like [RFDiffAA](https://github.com/baker-laboratory/rf_diffusion_all_atom) or others do not generate EC-class for the design of catalytic pockets. We use CLEAN to infer the EC-class of sequence representations of these pockets. For CLEAN, please refer to [CLEAN-official](https://github.com/tttianhao/CLEAN) or [CLEAN-webserver](https://clean.platform.moleculemaker.org/configuration). We use [CLEAN](https://github.com/tttianhao/CLEAN) with greedy ```max-separation``` approach for EC-class inference.

### 3. ESM3

For ESM3, please refer to [ESM3-official](https://github.com/evolutionaryscale/esm). For each sequence representation of generated catalytic pocket, we use [ESM3](https://github.com/evolutionaryscale/esm) to recover the full enzyme sequence (by 'entire' meaning, we recover 32 residues into a protein sequence of 200 residues). We can perform enzyme retrieval on both (1) pocket enzymes sequences and (2) full enzyme sequences.

### 4. Reaction-specified Enzyme Retrieval

For ranking-based retrieval evluation, please refer to [RectZyme-paper](https://www.arxiv.org/pdf/2408.13659). We train a enzyme retrieval model with enzyme features computed by [ESM2-650M](https://github.com/facebookresearch/esm) and [MAT-2D](https://github.com/ardigen/MAT). The training data are those of 60%-homology (~50,000 positive samples); [evaluation data](https://github.com/WillHua127/EnzymeFlow/blob/main/data/eval-data_cutoff-0.1_unique-subs-enz_100.csv) are those unique, non-repeated ones; training negative samples are training data that are not annotated to catalyze a specific reaction like [ClipZyme](https://arxiv.org/pdf/2402.06748); evaluation do not use negative data.



## Data Preparation
![pocket](./image/pocket.jpg)

### 1. Enzyme Pocket, Substrate Molecule, Product Molecule Rawdata
   
   $~~~~$ (a) [molecule_structures](https://github.com/WillHua127/EnzymeFlow/tree/main/data/molecule_structures) folder in ```./data``` contain all substrate and product molecules, can be downloaded at link.
   
   $~~~~$ (b) [pocket_fixed_residues/pdb_10A](https://github.com/WillHua127/EnzymeFlow/tree/main/data/pocket_fixed_residues/pdb_10A) folder in ```./data``` contain all enzyme pockets, can be downloaded at link.

   $~~~~$ (c) We provide [rawdata-40%homology](https://github.com/WillHua127/EnzymeFlow/blob/main/data/rawdata_cutoff-0.4.csv) and [metadata-40%homology](https://github.com/WillHua127/EnzymeFlow/blob/main/data/metadata_cutoff-0.4.csv) with 40% homologys in ```./data``` folder. More rawdata (50%, 60%, 80%, 90% homologys) can be downloaded at link.



### 2. Co-evolution and MSA
   
   $~~~~$ (a) [rxn_to_smiles_msa.pkl](https://github.com/WillHua127/EnzymeFlow/blob/main/data/rxn_to_smiles_msa.pkl) in ```./data``` contain reaction MSAs.
   
   $~~~~$ (b) [uid_to_protein_msa.pkl](link) in ```./data``` contain enzyme MSAs, can be downloaded at link.

   $~~~~$ (c) [vocab.txt](https://github.com/WillHua127/EnzymeFlow/blob/main/data/vocab.txt) in ```./data``` is co-evolution vocabulary.


### When the raw data--enzyme pockets, molecules, co-evolution--are ready (stored in right folders), we proceed to process them into metadata.

### 3. Process rawdata into metadata by running ```process_data.py```.
   
   $~~~~$ (a) Remember to change the configs ```--rawdata_file_name```, e.g., ```python process_data.py --rawdata_file_name rawdata_cutoff-0.4.csv```. Warning: we have absolute path in ```metadata.csv```, so you might need to change it to your path.

   
### 4. Processed Metadata.

   $~~~~$ (a) Processed metadata will be saved into ```./data/processed``` folder, including:
   
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

