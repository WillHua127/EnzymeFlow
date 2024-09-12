import io, os
import dataclasses

from typing import Dict, List, Tuple, Union, Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
import numpy as np

from Bio.PDB import PDBParser

from ofold.data import data_transforms
from ofold.np import residue_constants
from ofold.np.protein import Protein
from ofold.utils import rigid_utils

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.rdchem import BondType

from data.utils import *

CHAIN_FEATS = ["atom_positions", "aatype", "atom_mask", "residue_index", "b_factors"]
UNPADDED_FEATS = [
    "t",
    "rot_vectorfield_scaling",
    "trans_vectorfield_scaling",
    "t_seq",
    "t_struct",
]
RIGID_FEATS = ["rigids_0", "rigids_t"]
PAIR_FEATS = ["rel_rots"]
BONDS = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

def parse_chain_feats(chain_feats, mol_feats=None, scale_factor=1.0):
    ca_idx = residue_constants.atom_order["CA"]
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, ca_idx]
    bb_pos = chain_feats["atom_positions"][:, ca_idx]
    if mol_feats is not None:
        bb_center = mol_feats["ligand_com"]
    else:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-10)

    chain_feats["atom_positions_before_com"] = np.copy(chain_feats["atom_positions"])
    centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, ca_idx]
    chain_feats["bb_com"] = bb_center
    return chain_feats


def from_pdb_string(pdb_fh: str, chain_id: Optional[str] = None) -> Protein:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            if res.resname == 'HOH' or res_shortname == 'X':
                continue
            
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    # Chain IDs are usually characters so map these to ints.
    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )
    

def parse_pdb_feats(pdb_file: str, mol_feats=None, scale_factor=1.0):
    chain_prot = from_pdb_string(pdb_file)
    chain_dict = dataclasses.asdict(chain_prot)
    feat_dict = {x: chain_dict[x] for x in CHAIN_FEATS}
    return parse_chain_feats(feat_dict, mol_feats=mol_feats, scale_factor=scale_factor)
    

def _process_pdb(pdb_file_path: str, mol_feats=None):
    processed_feats = parse_pdb_feats(pdb_file_path, mol_feats=mol_feats)

    # Run through OpenFold data transforms.
    chain_feats = {
        "aatype": torch.tensor(processed_feats["aatype"]).long(),
        "all_atom_positions": torch.tensor(processed_feats["atom_positions"]).double(),
        "all_atom_mask": torch.tensor(processed_feats["atom_mask"]).double(),
    }
    chain_feats = data_transforms.atom37_to_frames(chain_feats)
    chain_feats = data_transforms.make_atom14_masks(chain_feats)
    chain_feats = data_transforms.make_atom14_positions(chain_feats)
    chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

    residx_atom14_to_atom37 = chain_feats["residx_atom14_to_atom37"].clone().detach()
    residx_atom13_mask = residx_atom14_to_atom37[:, 1:].clamp(max=1)
    vanilla_atom_feat = transform_vanilla_atoms(residx_atom14_to_atom37)
    vanilla_atom_feat[:, 1:] = vanilla_atom_feat[:, 1:] * residx_atom13_mask

    res_idx = processed_feats["residue_index"]
    # To speed up processing, only take necessary features
    final_feats = {
        "aatype": chain_feats["aatype"],
        "seq_idx": res_idx,
        "residx_atom14_to_atom37": vanilla_atom_feat,
        "residue_index": processed_feats["residue_index"],
        "res_mask": processed_feats["bb_mask"],
        "atom37_pos": chain_feats["all_atom_positions"],
        "atom37_mask": chain_feats["all_atom_mask"],
        "atom14_pos": chain_feats["atom14_gt_positions"],
        "rigidgroups_1": chain_feats["rigidgroups_gt_frames"],
        "torsion_angles_sin_cos": chain_feats["torsion_angles_sin_cos"],
        "complex_com": processed_feats["bb_com"],
        "atom37_pos_before_com": processed_feats["atom_positions_before_com"],
    }

    return final_feats


def process_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)        
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        
    atom_feat = []
    for atom in mol.GetAtoms():
        atom_feat.append(atom.GetAtomicNum())
    
    rows, cols, edge_types = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        rows += [start, end]
        cols += [end, start]
        edge_types += 2 * [BONDS[bond.GetBondType()]]
    
    edge_index = [rows, cols]
    atom_feat = np.array(atom_feat)
    edge_index = np.array(edge_index)
    edge_types = np.array(edge_types)
    
    perm = (edge_index[0] * atom_feat.size + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_types = edge_types[perm]
    
    final_feats = {
        "molecule_atom_feat": atom_feat,
        "molecule_edge_idx": edge_index,
        "molecule_edge_feat": edge_types,
    }
    
    return final_feats
    
    

def process_mol(mol_file_path: str, protein_bb_center=None):
    mol = Chem.MolFromMol2File(mol_file_path)
    if mol is None:
        mol = Chem.MolFromMol2File(mol_file_path, sanitize=False)
    conf = mol.GetConformer()

    ligand_atom_feat = []
    ligand_atom_coord = []
    for atom in mol.GetAtoms():
        positions = conf.GetAtomPosition(atom.GetIdx())
        ligand_atom_feat.append(atom.GetAtomicNum())
        ligand_atom_coord.append([positions.x, positions.y, positions.z])

    ligand_atom_feat = np.array(ligand_atom_feat)
    ligand_atom_coord = np.array(ligand_atom_coord)
    if protein_bb_center is not None:
        ligand_atom_com = protein_bb_center
    else:
        ligand_atom_com = np.sum(ligand_atom_coord, axis=0) / (np.sum(ligand_atom_feat.clip(max=1.)) + 1e-10)

    ligand_pos_after_com = ligand_atom_coord - ligand_atom_com
    final_feats = {
        "ligand_feat": ligand_atom_feat,
        "ligand_pos": ligand_atom_coord,
        "ligand_com": ligand_atom_com,
        "ligand_pos_after_com": ligand_pos_after_com,
    }
    
    return final_feats


def process_pdb(pdb_file_path: str, mol_feats=None):
    chain_feats = _process_pdb(pdb_file_path, mol_feats=mol_feats)
    return chain_feats
