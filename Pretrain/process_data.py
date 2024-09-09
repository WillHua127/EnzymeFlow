import os
import random
import torch

from tqdm import tqdm
import pandas as pd
import argparse

from data.data import *
from data.utils import *
from ofold.np import residue_constants

parser = argparse.ArgumentParser(description='processing script.')
parser.add_argument(
    '--pdb_ligand_dir',
    type=str,
    default='./data/pdb')

parser.add_argument(
    '--meta_dir',
    type=str,
    default='./data')

parser.add_argument(
    '--label_file_name',
    type=str,
    default='kdvalue.csv')

parser.add_argument(
    '--metadata_file_name',
    type=str,
    default='metadata_all.csv')


CA_IDX = residue_constants.atom_order["CA"]

def reshape_protein(protein, mask):
    _protein = {}
    for k, v in protein.items():
        if k not in {"complex_com"}:
            _protein[k] = v[mask]
        else:
            _protein[k] = v
    
    return _protein 

def process_metadata(meta_dir, pocket_file_path, ligand_file_path, pdb_name, kd_value):
    metadata = {}
    #processing
    protein = process_pdb(pocket_file_path, mol_feats=None)
    ligand = process_mol(ligand_file_path, protein_bb_center=protein["complex_com"])

    if kd_value<2:
        kd_range = 1
    elif kd_value>=2 and kd_value<3:
        kd_range = 2
    elif kd_value>=3 and kd_value<4:
        kd_range = 3
    elif kd_value>=4 and kd_value<5:
        kd_range = 4
    elif kd_value>=5 and kd_value<6:
        kd_range = 5
    elif kd_value>=6 and kd_value<7:
        kd_range = 6
    elif kd_value>=7 and kd_value<8:
        kd_range = 7
    elif kd_value>=8 and kd_value<9:
        kd_range = 8
    elif kd_value>=9 and kd_value<10:
        kd_range = 9
    elif kd_value>=10:
        kd_range = 10

    #saving metadata
    metadata["pdb_name"] = pdb_name
    metadata["raw_protein_path"] = pocket_file_path
    metadata["raw_ligand_path"] = ligand_file_path
    metadata["num_ligand_atom"] = len(ligand["ligand_feat"])
    metadata["num_protein_amino_acid"] = len(protein["aatype"])
    metadata["num_protein_atom"] = int(protein['atom37_mask'].sum(dim=(0,1)).item())
    metadata["kd_value"] = kd_value
    metadata["kd_class"] = kd_range

    meta_subdir = os.path.join(meta_dir, pdb_name.lower())
    if not os.path.isdir(meta_subdir):
        os.mkdir(meta_subdir)
    processed_meta_path = os.path.join(meta_subdir, f"{pdb_name}.pkl")
    processed_meta_path = os.path.abspath(processed_meta_path)
    metadata["processed_path"] = processed_meta_path

    complex_feats = {
        "ligand": ligand, 
        "protein": protein,
    }

    write_pkl(processed_meta_path, complex_feats)
    
    return metadata
    

def get_all_files(pdb_ligand_dir):
    all_pdb_ligand_names = os.listdir(pdb_ligand_dir)
    all_pdb_ligand_files = [{"protein_file": os.path.join(pdb_ligand_dir, pdb_ligand_name, pdb_ligand_name+"_pocket.pdb"), 
                             "ligand_file": os.path.join(pdb_ligand_dir, pdb_ligand_name, pdb_ligand_name+"_ligand.mol2"),
                             "pdb_name": pdb_ligand_name,}
                            for pdb_ligand_name in all_pdb_ligand_names]
    return all_pdb_ligand_files
    


def main(args):
    pdb_ligand_dir = args.pdb_ligand_dir
    meta_dir = args.meta_dir
    label_file_name = args.label_file_name
    metadata_file_name = args.metadata_file_name


    all_pdb_ligand_files = get_all_files(pdb_ligand_dir)

    if not os.path.isdir(meta_dir):
        os.makedirs(meta_dir, exist_ok=True)

    label_path = os.path.join(args.meta_dir, label_file_name)
    label_file = pd.read_csv(label_path, index_col='PDB')['-logKd/Ki'].to_dict()

    metadata_path = os.path.join(args.meta_dir, metadata_file_name)
    print(f'Files will be written to {metadata_path}')

    meta_processed_dir = os.path.join(args.meta_dir, "processed")
    if not os.path.isdir(meta_processed_dir):
        os.makedirs(meta_processed_dir, exist_ok=True)

    all_metadata = []
    for pdb_ligand_file in tqdm(all_pdb_ligand_files):
        try:
            pocket_file_path = pdb_ligand_file["protein_file"]
            ligand_file_path = pdb_ligand_file["ligand_file"]
            pdb_name = pdb_ligand_file["pdb_name"]
            kd_value = float(label_file[pdb_name])

            metadata = process_metadata(meta_processed_dir, pocket_file_path, ligand_file_path, pdb_name, kd_value)
            all_metadata.append(metadata)
    
        except:
            print(f"Failed process {pocket_file_path} {ligand_file_path} {pdb_name}")
    
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(f'Finished processing {succeeded}/{len(all_pdb_ligand_files)} files')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
