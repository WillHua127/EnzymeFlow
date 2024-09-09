import io, os
import re
import pickle

import dataclasses

from typing import Dict, List, Tuple, Union, Any, Mapping, Optional, Sequence

import torch
import numpy as np

from ofold.np.protein import Protein, to_pdb

def transform_vanilla_atoms(atom_feat):
    feat = atom_feat.clone()
    feat[feat==0] = 40
    feat[feat==1] = 41
    feat[feat==2] = 41
    feat[feat==3] = 41
    feat[feat==4] = 42
    feat[feat==5] = 41
    feat[feat==6] = 41
    feat[feat==7] = 41
    feat[feat==8] = 42
    feat[feat==9] = 42
    feat[feat==10] = 43
    feat[feat==11] = 41
    feat[feat==12] = 41
    feat[feat==13] = 41
    feat[feat==14] = 40
    feat[feat==15] = 40
    feat[feat==16] = 42
    feat[feat==17] = 42
    feat[feat==18] = 43
    feat[feat==19] = 41
    feat[feat==20] = 41
    feat[feat==21] = 41
    feat[feat==22] = 41
    feat[feat==23] = 40
    feat[feat==24] = 40
    feat[feat==25] = 40
    feat[feat==26] = 42
    feat[feat==27] = 42
    feat[feat==28] = 41
    feat[feat==29] = 40
    feat[feat==30] = 40
    feat[feat==31] = 42
    feat[feat==32] = 41
    feat[feat==33] = 41
    feat[feat==34] = 41
    feat[feat==35] = 40
    feat[feat==36] = 42
    
    #N=7, C=6, O=8, S=16
    feat[feat==40] = 7
    feat[feat==41] = 6
    feat[feat==42] = 8
    feat[feat==43] = 16
    return feat


def create_full_prot(
    atom14: np.ndarray,
    atom14_mask: np.ndarray,
    aatype=None,
    b_factors=None,
):
    assert atom14.ndim == 3
    assert atom14.shape[-1] == 3
    assert atom14.shape[-2] == 37
    n = atom14.shape[0]
    residue_index = np.arange(n)
    chain_index = np.zeros(n)
    if b_factors is None:
        b_factors = np.zeros([n, 37])
    if aatype is None:
        aatype = np.zeros(n, dtype=int)
    return Protein(
        atom_positions=atom14,
        atom_mask=atom14_mask,
        aatype=aatype,
        residue_index=residue_index,
        chain_index=chain_index,
        b_factors=b_factors,
    )
    

def write_pkl(
        save_path: str, pkl_data: Any, create_dir: bool = False, use_torch=False):
    """Serialize data into a pickle file."""
    if create_dir:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if use_torch:
        torch.save(pkl_data, save_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(save_path, 'wb') as handle:
            pickle.dump(pkl_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


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
            

def write_prot_to_pdb(
    prot_pos: np.ndarray,
    file_path: str,
    aatype: np.ndarray = None,
    overwrite=False,
    no_indexing=False,
    b_factors=None,
):
    if overwrite:
        max_existing_idx = 0
    else:
        file_dir = os.path.dirname(file_path)
        file_name = os.path.basename(file_path).strip(".pdb")
        existing_files = [x for x in os.listdir(file_dir) if file_name in x]
        max_existing_idx = max(
            [
                int(re.findall(r"_(\d+).pdb", x)[0])
                for x in existing_files
                if re.findall(r"_(\d+).pdb", x)
                if re.findall(r"_(\d+).pdb", x)
            ]
            + [0]
        )
    if not no_indexing:
        save_path = file_path.replace(".pdb", "") + f"_{max_existing_idx+1}.pdb"
    else:
        save_path = file_path
    with open(save_path, "w") as f:
        if prot_pos.ndim == 4:
            for t, pos37 in enumerate(prot_pos):
                atom37_mask = np.sum(np.abs(pos37), axis=-1) > 1e-7
                prot = create_full_prot(
                    pos37, atom37_mask, aatype=aatype, b_factors=b_factors
                )
                pdb_prot = protein.to_pdb(prot, model=t+1, add_end=False)
                f.write(pdb_prot)
        elif prot_pos.ndim == 3:
            atom37_mask = np.sum(np.abs(prot_pos), axis=-1) > 1e-7
            prot = create_full_prot(
                prot_pos, atom37_mask, aatype=aatype, b_factors=b_factors
            )
            pdb_prot = to_pdb(prot, model=1, add_end=False)
            f.write(pdb_prot)
        else:
            raise ValueError(f"Invalid positions shape {prot_pos.shape}")
        f.write("END")
    return save_path
