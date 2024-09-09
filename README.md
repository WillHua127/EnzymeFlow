# EnzymeFlow

## Model
![enzymeflow](./image/enzymeflow.jpg)


## Data
![pocket](./image/pocket.jpg)

1. Enzyme Pocket and Substrate, Product Molecule Rawdata
   
   (a) [molecule_structures folder](https://github.com/WillHua127/EnzymeFlow/tree/main/data/molecule_structures) in ./data contain all substrate and product molecules, can be downloaded at link.
   
   (b) [pocket_fixed_residues/pdb_10A folder](https://github.com/WillHua127/EnzymeFlow/tree/main/data/pocket_fixed_residues/pdb_10A) in ./data contain all enzyme pockets, can be downloaded at link.

   (c) In this github, we provide [rawdata](https://github.com/WillHua127/EnzymeFlow/blob/main/data/rawdata_cutoff-0.4.csv) and [metadata](https://github.com/WillHua127/EnzymeFlow/blob/main/data/metadata_cutoff-0.4.csv) with 40% homologys in ./data folder. More rawdata can be downloaded at link.



2. Co-evolution and MSA
   
   (a) [rxn_to_smiles_msa.pkl](https://github.com/WillHua127/EnzymeFlow/blob/main/data/rxn_to_smiles_msa.pkl) in ./data contain reaction MSAs.
   
   (b) [uid_to_protein_msa.pkl](link) in ./data contain enzyme MSAs, can be downloaded at link.

   (c) [vocab.txt](https://github.com/WillHua127/EnzymeFlow/blob/main/data/vocab.txt) in ./data is co-evolution vocabulary.


### When the raw enzyme pockets, molecules, co-evolution data are ready, then

3. Process rawdata into metadata by running process_data.py.
   
   (a) Remember to change the configs --rawdata_file_name, e.g., --rawdata_file_name rawdata_cutoff-0.4.csv.

    Warning: we have absolute path in metadata.csv, so you might need to change it to your path.

   
4. Processed Metadata.

   (a) Processed data will be saved into ./data/processed folder, with:
   
   (b) processed enzyme in ./data/processed/protein

   (c) processed substrate in ./data/processed/ligand

   (d) processed co-evolution in ./data/processed/msa

   (e) processed produuct in ./data/processed/product

   (f) a toy [example](https://github.com/WillHua127/EnzymeFlow/tree/main/data/processed) is provided.






## Further Statistics
![distribution](./image/distribution.jpg)

