import os
import random
import torch

from tqdm import tqdm
import pandas as pd
import argparse

from data.data import *
from data.utils import *


parser = argparse.ArgumentParser(description='processing script.')
parser.add_argument(
    '--pdb_dir',
    type=str,
    default='./data/pocket_fixed_residues/pdb_10A')

parser.add_argument(
    '--mol_dir',
    type=str,
    default='./data/moleucle_structures')

parser.add_argument(
    '--meta_dir',
    type=str,
    default='./data')

parser.add_argument(
    '--vocab_file_name',
    type=str,
    default='vocab.txt')

parser.add_argument(
    '--reaction_msa_file_name',
    type=str,
    default='rxn_to_smiles_msa.pkl')

parser.add_argument(
    '--enzyme_msa_file_name',
    type=str,
    default='uid_to_protein_msa.pkl')

parser.add_argument(
    '--rawdata_file_name',
    type=str,
    default='rawdata.csv')

parser.add_argument(
    '--metadata_file_name',
    type=str,
    default='metadata.csv')

RXN_GAP_CHAR = ' '
ENZYME_GAP_CHAR = '-'

def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, "rb") as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, "rb") as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(
                    f"Failed to read {read_path}. First error: {e}\n Second error: {e2}"
                )
            raise (e)

def cut_protein(protein, n_res=32):
    cut_protein = {}
    for k, v in protein.items():
        cut_protein[k] = v[:n_res, ...]
    
    return cut_protein 
    

def token2id(text, vocab):
    unknown_token_id = vocab.get('<unk>')
    data = []
    for sentence in text:
        if isinstance(sentence, str):
            sentence = list(sentence)
        sentence = [vocab.get(s, unknown_token_id) for s in sentence]
        data.append(sentence)
    return data

def remove_all_dash_columns(alignment, gap_char):
    # remove columns whose content are all gaps
    parsed_alignment = [list(line) for line in alignment]
    transposed = list(zip(*parsed_alignment))
    filtered_transposed = [col for col in transposed if any(char != gap_char for char in col)]    
    filtered_alignment = [''.join(col) for col in zip(*filtered_transposed)]
    return filtered_alignment
    

def process_msa(name, msa_feature, vocab, gap_char):
    msa = msa_feature[name]
    msa = remove_all_dash_columns(msa, gap_char)
    token_msa = token2id(msa, vocab)
    token_msa = torch.tensor(token_msa)
    return token_msa
    
    

def process_metadata(meta_dir, pdb_dir, mol_dir, rawdata, reaction_msa, enzyme_msa, vocab):
    metadata = {}

    #pdb_name = str(rawdata["pdb_name"])
    pdb_name = str(rawdata["uniprotID"])
    reaction_smiles = str(rawdata["CANO_RXN_SMILES"])
    product_smiles = str(rawdata["Product"])
    product_id = str(rawdata["Product_ID"])
    substrate_id = str(rawdata["Substrate_ID"])
    ec_class = int(rawdata["ec_class"])
    ligand=None
    protein_bb_center=None
    
    
    meta_substrate_subdir = os.path.join(meta_dir, "ligand", substrate_id)
    #ligand_file_path = os.path.join(mol_dir, substrate_id, f"{substrate_id}_ligand.mol2")
    ligand_file_path = os.path.join(mol_dir, f"{substrate_id}.mol2")
    processed_meta_ligand_path = os.path.join(meta_substrate_subdir, f"{substrate_id}.pkl")
    processed_meta_ligand_path = os.path.abspath(processed_meta_ligand_path)
    metadata["processed_ligand_path"] = processed_meta_ligand_path
    if not os.path.exists(processed_meta_ligand_path):
        ligand = process_mol(ligand_file_path, protein_bb_center=protein_bb_center)
        if not os.path.isdir(meta_substrate_subdir):
            os.mkdir(meta_substrate_subdir)
        write_pkl(processed_meta_ligand_path, {"ligand": ligand})

    else:
        ligand = read_pkl(processed_meta_ligand_path)["ligand"]
    
    
    meta_enzyme_subdir = os.path.join(meta_dir, "protein", pdb_name.lower())
    #pocket_file_path = os.path.join(pdb_dir, pdb_name, f"{pdb_name}_pocket.pdb")
    pocket_file_path = os.path.join(pdb_dir, f"{pdb_name}.pdb")
    processed_meta_protein_path = os.path.join(meta_enzyme_subdir, f"{pdb_name}.pkl")
    processed_meta_protein_path = os.path.abspath(processed_meta_protein_path)
    metadata["processed_protein_path"] = processed_meta_protein_path
    if not os.path.exists(processed_meta_protein_path):
        if not os.path.isdir(meta_enzyme_subdir):
            os.mkdir(meta_enzyme_subdir)
        protein = process_pdb(pocket_file_path, mol_feats=None)
        protein = cut_protein(protein) #not necessary

        write_pkl(processed_meta_protein_path, {"protein": protein})
        protein_bb_center = protein["complex_com"]

    else:
        protein = read_pkl(processed_meta_protein_path)["protein"]
        protein_bb_center = protein["complex_com"]

    
    
    meta_product_subdir = os.path.join(meta_dir, "product", product_id)
    processed_meta_product_path = os.path.join(meta_product_subdir, f"{product_id}.pkl")
    processed_meta_product_path = os.path.abspath(processed_meta_product_path)
    metadata["processed_product_path"] = processed_meta_product_path
    if not os.path.exists(processed_meta_product_path):
        product = process_smiles(product_smiles)
        if not os.path.isdir(meta_product_subdir):
            os.mkdir(meta_product_subdir)
        write_pkl(processed_meta_product_path, {"product": product})
        

    
    meta_reaction_msa_subdir = os.path.join(meta_dir, "msa", f"{substrate_id}_{product_id}")
    processed_meta_reaction_msa_path = os.path.join(meta_reaction_msa_subdir, f"{substrate_id}_{product_id}.pkl")
    processed_meta_reaction_msa_path = os.path.abspath(processed_meta_reaction_msa_path)
    metadata["processed_reaction_msa_path"] = processed_meta_reaction_msa_path
    if not os.path.exists(processed_meta_reaction_msa_path):
        r_msa = process_msa(reaction_smiles, reaction_msa, vocab, RXN_GAP_CHAR)
        if not os.path.isdir(meta_reaction_msa_subdir):
            os.mkdir(meta_reaction_msa_subdir)
        write_pkl(processed_meta_reaction_msa_path, {"reaction_msa": r_msa})


    meta_enzyme_msa_subdir = os.path.join(meta_dir, "msa", f"{pdb_name}")
    processed_meta_enzyme_msa_path = os.path.join(meta_enzyme_msa_subdir, f"{pdb_name}.pkl")
    processed_meta_enzyme_msa_path = os.path.abspath(processed_meta_enzyme_msa_path)
    metadata["processed_enzyme_msa_path"] = processed_meta_enzyme_msa_path
    if not os.path.exists(processed_meta_enzyme_msa_path):
        e_msa = process_msa(pdb_name, enzyme_msa, vocab, ENZYME_GAP_CHAR)
        if not os.path.isdir(meta_enzyme_msa_subdir):
            os.mkdir(meta_enzyme_msa_subdir)
        write_pkl(processed_meta_enzyme_msa_path, {"enzyme_msa": e_msa})
    

    #saving metadata
    metadata["pdb_name"] = pdb_name
    metadata["substrate_name"] = substrate_id
    metadata["product_name"] = product_id
    metadata["raw_protein_path"] = pocket_file_path
    metadata["raw_ligand_path"] = ligand_file_path
    metadata["num_ligand_atom"] = len(ligand["ligand_feat"])
    metadata["num_protein_amino_acid"] = len(protein["aatype"])
    metadata["num_protein_atom"] = protein['atom37_mask'].sum(dim=(0,1)).long().item()

    # metadata["num_msa"] = 5
    metadata["ec_class"] = ec_class
    metadata["product_smiles"] = product_smiles
    metadata["reaction_smiles"] = reaction_smiles
    
    return metadata
    

def read_pkl(read_path: str, verbose=True, use_torch=False, map_location=None):
    """Read data from a pickle file."""
    try:
        if use_torch:
            return torch.load(read_path, map_location=map_location)
        else:
            with open(read_path, 'rb') as handle:
                return pickle.load(handle)
    except Exception as e:
        try:
            with open(read_path, 'rb') as handle:
                return CPU_Unpickler(handle).load()
        except Exception as e2:
            if verbose:
                print(f'Failed to read {read_path}. First error: {e}\n Second error: {e2}')
            raise(e)
            

def load_vocab(vocab_data):
    if isinstance(vocab_data, dict):
        return vocab_data
    elif isinstance(vocab_data, str) and os.path.exists(vocab_data):
        if vocab_data.endswith('txt'):
            with open(vocab_data, 'r') as f:
                vocab = [each.strip('\n') for each in f.readlines()]
                vocab = {token: i for i, token in enumerate(vocab)}
        elif vocab_data.endswith('pkl'):
            with open(vocab_data, 'rb') as f:
                vocab = pkl.load(f)
        else:
            raise ValueError(f'Unsupported vocab file format: {vocab_data}')
    return vocab
    


def main(args):
    pdb_dir = args.pdb_dir
    mol_dir = args.mol_dir
    meta_dir = args.meta_dir
    rawdata_file_name = args.rawdata_file_name
    metadata_file_name = args.metadata_file_name
    
    vocab_file_name = args.vocab_file_name
    reaction_msa_file_name = args.reaction_msa_file_name
    enzyme_msa_file_name = args.enzyme_msa_file_name


    if not os.path.isdir(meta_dir):
        os.makedirs(meta_dir, exist_ok=True)

    rawdata_path = os.path.join(args.meta_dir, rawdata_file_name)
    rawdata_file = pd.read_csv(rawdata_path)
    n_data = len(rawdata_file)

    metadata_path = os.path.join(args.meta_dir, metadata_file_name)
    print(f'Files will be written to {metadata_path}')

    vocab_path = os.path.join(args.meta_dir, vocab_file_name)
    vocab = load_vocab(vocab_path)
    
    reaction_msa_path = os.path.join(args.meta_dir, reaction_msa_file_name)
    enzyme_msa_path = os.path.join(args.meta_dir, enzyme_msa_file_name)
    reaction_msa = read_pkl(reaction_msa_path)
    enzyme_msa = read_pkl(enzyme_msa_path)
    
    meta_processed_dir = os.path.join(args.meta_dir, "processed")
    if not os.path.isdir(meta_processed_dir):
        os.makedirs(meta_processed_dir, exist_ok=True)

    meta_ligand_processed_dir = os.path.join(meta_processed_dir, "ligand")
    if not os.path.isdir(meta_ligand_processed_dir):
        os.makedirs(meta_ligand_processed_dir, exist_ok=True)

    meta_protein_processed_dir = os.path.join(meta_processed_dir, "protein")
    if not os.path.isdir(meta_protein_processed_dir):
        os.makedirs(meta_protein_processed_dir, exist_ok=True)

    meta_product_processed_dir = os.path.join(meta_processed_dir, "product")
    if not os.path.isdir(meta_product_processed_dir):
        os.makedirs(meta_product_processed_dir, exist_ok=True)

    meta_msa_processed_dir = os.path.join(meta_processed_dir, "msa")
    if not os.path.isdir(meta_msa_processed_dir):
        os.makedirs(meta_msa_processed_dir, exist_ok=True)

    all_metadata = []
    for idx in tqdm(range(n_data)):
        try:
            rawdata = rawdata_file.iloc[idx]
            metadata = process_metadata(meta_processed_dir, pdb_dir, mol_dir, rawdata, reaction_msa, enzyme_msa, vocab)
            all_metadata.append(metadata)
    
        except:
            name = rawdata["uniprotID"]
            print(f"Failed process {idx} {name}")
    
    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(f'Finished processing {succeeded}/{n_data} files')


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
