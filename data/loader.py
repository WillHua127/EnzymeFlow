import io, os
import logging
import random

import pandas as pd

import torch
import numpy as np
import torch.nn.functional as F

import ot as pot
from scipy.spatial.transform import Rotation

from typing import Dict, List, Tuple, Union, Any, Mapping, Optional, Sequence

from flowmatch.utils.rigid_helpers import assemble_rigid_mat, extract_trans_rots_mat
from flowmatch.utils.so3_helpers import so3_relative_angle
from flowmatch.data import utils as du

from ofold.utils import rigid_utils as ru

LIGAND_PADDING_FEATS = ["ligand_atom", "ligand_pos", "ligand_mask"]
GUIDE_LIGAND_PADDING_ATOM_FEATS = ["guide_ligand_atom", "guide_ligand_atom_mask"]
GUIDE_LIGAND_PADDING_EDGE_FEATS = ["guide_ligand_edge", "guide_ligand_edge_index", "guide_ligand_edge_mask"]
MSA_PADDING_FEATS = ["msa_1", "msa_mask", "msa_onehot_1", "msa_vectorfield", "msa_onehot_0", "msa_onehot_t", "msa_t"]

def pad(x: np.ndarray, max_len: int, pad_idx=0, use_torch=False, reverse=False):
    # Pad only the residue dimension.
    seq_len = x.shape[pad_idx]
    pad_amt = max_len - seq_len
    pad_widths = [(0, 0)] * x.ndim
    if pad_amt < 0:
        raise ValueError(f"Invalid pad amount {pad_amt}")
    if reverse:
        pad_widths[pad_idx] = (pad_amt, 0)
    else:
        pad_widths[pad_idx] = (0, pad_amt)
    if use_torch:
        return torch.pad(x, pad_widths)
    return np.pad(x, pad_widths)
    
def pad_feats(raw_feats, max_len, pad_msa=False, pad_guide_atom=False, pad_guide_edge=False, use_torch=False):
    if pad_msa:
        PADDING_FEATS = MSA_PADDING_FEATS

    elif pad_guide_atom:
        PADDING_FEATS = GUIDE_LIGAND_PADDING_ATOM_FEATS

    elif pad_guide_edge:
        PADDING_FEATS = GUIDE_LIGAND_PADDING_EDGE_FEATS

    else:
        PADDING_FEATS = LIGAND_PADDING_FEATS
        

    if pad_guide_edge:
        if "guide_ligand_edge_index" in raw_feats.keys():
            raw_feats["guide_ligand_edge_index"] = raw_feats["guide_ligand_edge_index"].transpose(0,1)
        
    padded_feats = {
        feat_name: pad(feat, max_len, use_torch=use_torch)
        for feat_name, feat in raw_feats.items()
        if feat_name in PADDING_FEATS
    }

    if pad_guide_edge:
        if "guide_ligand_edge_index" in padded_feats.keys():
            padded_feats["guide_ligand_edge_index"] = padded_feats["guide_ligand_edge_index"].transpose(1,0)

    for feat_name in raw_feats:
        if feat_name not in PADDING_FEATS:
            padded_feats[feat_name] = raw_feats[feat_name]
        else:
            padded_feats[feat_name] = padded_feats[feat_name]
            
    return padded_feats


def length_batching_multi_gpu(
    feat_dicts,
    num_gpus: int,
):
    def get_protein_len(x):
        return x["res_mask"].shape[0]

    def get_ligand_len(x):
        return x["ligand_mask"].shape[0]

    def get_guide_ligand_atom_len(x):
        return x["guide_ligand_atom_mask"].shape[0]

    def get_guide_ligand__edge_len(x):
        return x["guide_ligand_edge_mask"].shape[0]

    def get_msa_len(x):
        return x["msa_t"].shape[0]


    feat_dicts = [x for x in feat_dicts if x is not None]
    dicts_by_ligand_length = [(get_ligand_len(x), x) for x in feat_dicts]
    ligand_length_sorted = sorted(dicts_by_ligand_length, key=lambda x: x[0], reverse=True)
    max_ligand_len = ligand_length_sorted[0][0]
    pad_example = lambda x: pad_feats(x, max_ligand_len)
    padded_batch = [pad_example(x) for (_, x) in ligand_length_sorted]
    

    # pad guide molecule
    dicts_by_guide_ligand_atom_length = [(get_guide_ligand_atom_len(x), x) for x in padded_batch]
    guide_ligand_atom_length_sorted = sorted(dicts_by_guide_ligand_atom_length, key=lambda x: x[0], reverse=True)
    max_guide_ligand_atom_len = guide_ligand_atom_length_sorted[0][0]
    pad_guide_atom_example = lambda x: pad_feats(x, max_guide_ligand_atom_len, pad_guide_atom=True)
    padded_batch = [pad_guide_atom_example(x) for (_, x) in guide_ligand_atom_length_sorted]
    

    dicts_by_guide_ligand_edge_length = [(get_guide_ligand__edge_len(x), x) for x in padded_batch]
    guide_ligand_edge_length_sorted = sorted(dicts_by_guide_ligand_edge_length, key=lambda x: x[0], reverse=True)
    max_guide_ligand_edge_len = guide_ligand_edge_length_sorted[0][0]
    pad_guide_edge_example = lambda x: pad_feats(x, max_guide_ligand_edge_len, pad_guide_edge=True)
    padded_batch = [pad_guide_edge_example(x) for (_, x) in guide_ligand_edge_length_sorted]
    

    # pad msa dim=0
    if "msa_t" in feat_dicts[0].keys():
        dicts_by_msa_num = [(get_msa_len(x), x) for x in padded_batch]
        msa_num_sorted = sorted(dicts_by_msa_num, key=lambda x: x[0], reverse=True)
        max_msa_num = msa_num_sorted[0][0]
        pad_msa_example = lambda x: pad_feats(x, max_msa_num, pad_msa=True)
        padded_batch = [pad_msa_example(x) for (_, x) in msa_num_sorted]
        
    return torch.utils.data.default_collate(padded_batch)
    

def length_batching(
    feat_dicts,
):
    def get_protein_len(x):
        return x["res_mask"].shape[0]

    def get_ligand_len(x):
        return x["ligand_mask"].shape[0]

    def get_guide_ligand_atom_len(x):
        return x["guide_ligand_atom_mask"].shape[0]

    def get_guide_ligand__edge_len(x):
        return x["guide_ligand_edge_mask"].shape[0]

    def get_msa_len(x):
        return x["msa_t"].shape[0]


    feat_dicts = [x for x in feat_dicts if x is not None]
    dicts_by_ligand_length = [(get_ligand_len(x), x) for x in feat_dicts]
    ligand_length_sorted = sorted(dicts_by_ligand_length, key=lambda x: x[0], reverse=True)
    max_ligand_len = ligand_length_sorted[0][0]
    pad_example = lambda x: pad_feats(x, max_ligand_len)
    padded_batch = [pad_example(x) for (_, x) in ligand_length_sorted]
    

    if "guide_ligand_atom_mask" in feat_dicts[0].keys():
        # pad guide molecule
        dicts_by_guide_ligand_atom_length = [(get_guide_ligand_atom_len(x), x) for x in padded_batch]
        guide_ligand_atom_length_sorted = sorted(dicts_by_guide_ligand_atom_length, key=lambda x: x[0], reverse=True)
        max_guide_ligand_atom_len = guide_ligand_atom_length_sorted[0][0]
        pad_guide_atom_example = lambda x: pad_feats(x, max_guide_ligand_atom_len, pad_guide_atom=True)
        padded_batch = [pad_guide_atom_example(x) for (_, x) in guide_ligand_atom_length_sorted]

    if "guide_ligand_edge_mask" in feat_dicts[0].keys():
        dicts_by_guide_ligand_edge_length = [(get_guide_ligand__edge_len(x), x) for x in padded_batch]
        guide_ligand_edge_length_sorted = sorted(dicts_by_guide_ligand_edge_length, key=lambda x: x[0], reverse=True)
        max_guide_ligand_edge_len = guide_ligand_edge_length_sorted[0][0]
        pad_guide_edge_example = lambda x: pad_feats(x, max_guide_ligand_edge_len, pad_guide_edge=True)
        padded_batch = [pad_guide_edge_example(x) for (_, x) in guide_ligand_edge_length_sorted]
    

    # pad msa dim=0
    if "msa_t" in feat_dicts[0].keys():
        dicts_by_msa_num = [(get_msa_len(x), x) for x in padded_batch]
        msa_num_sorted = sorted(dicts_by_msa_num, key=lambda x: x[0], reverse=True)
        max_msa_num = msa_num_sorted[0][0]
        pad_msa_example = lambda x: pad_feats(x, max_msa_num, pad_msa=True)
        padded_batch = [pad_msa_example(x) for (_, x) in msa_num_sorted]
        
    return torch.utils.data.default_collate(padded_batch)


def possible_tuple_length_batching_multi_gpu(
    x,
    num_gpus: int,
):
    if type(x[0]) == tuple:
        return length_batching_multi_gpu(
            [y[0] for y in x], num_gpus
        ), [y[1] for y in x]
    else:
        return length_batching_multi_gpu(x, num_gpus)
        
def possible_tuple_length_batching(
    x,
):
    if type(x[0]) == tuple:
        return length_batching([y[0] for y in x]), [y[1] for y in x]
    else:
        return length_batching(x)


def create_data_loader(
    torch_dataset: torch.utils.data.Dataset,
    batch_size,
    shuffle,
    sampler=None,
    num_workers=0,
    length_batch=False,
    drop_last=False,
    prefetch_factor=2,
    num_gpus=1,
):
    if length_batch:
        if num_gpus > 1:
            collate_fn = lambda x: possible_tuple_length_batching_multi_gpu(
                x, num_gpus=num_gpus,
            )
        else:
            collate_fn = lambda x: possible_tuple_length_batching(
                x,
            )
    else:
        collate_fn = None

    persistent_workers = True if num_workers > 0 else False
    return torch.utils.data.DataLoader(
                    torch_dataset,
                    sampler=sampler,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    collate_fn=collate_fn,
                    num_workers=num_workers,
                    persistent_workers=persistent_workers,
                    pin_memory=True,
                    drop_last=drop_last,
                    multiprocessing_context="fork" if num_workers != 0 else None,
                )
        

def get_csv_row(args, csv, idx):
    """Get on row of the csv file, and prepare the pdb feature dict.

    Args:
        idx (int): idx of the row
        csv (pd.DataFrame): csv pd.DataFrame

    Returns:
        tuple: dict of the features, ground truth backbone rigid, pdb_name
    """
    # Sample data example.
    example_idx = idx
    csv_row = csv.iloc[example_idx]
    
    pdb_name = str(csv_row["pdb_name"])
    substrate_name = str(csv_row["substrate_name"])
    product_name = str(csv_row["product_name"])
        
    identifier = f"{pdb_name}_{substrate_name}_{product_name}"
    
        
    protein_path, ligand_path = csv_row["processed_protein_path"], csv_row["processed_ligand_path"]
    chain_feats, ligand_feats = du.read_pkl(protein_path)["protein"], du.read_pkl(ligand_path)["ligand"]

    # processed_feats = du.read_pkl(csv_row["processed_path"])
    # chain_feats, ligand_feats = processed_feats["protein"], processed_feats["ligand"]

    # process protein
    gt_bb_rigid = ru.Rigid.from_tensor_4x4(chain_feats["rigidgroups_1"])[:, 0]
    flowed_mask = np.ones_like(chain_feats["res_mask"])
    flow_mask = np.ones_like(chain_feats["res_mask"])
    chain_feats["flow_mask"] = flow_mask
    chain_feats["rigids_1"] = gt_bb_rigid.to_tensor_7()
    chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())
    chain_feats["sc_aa_t"] = torch.zeros(flow_mask.size, args.num_aa_type)
    # flowed_mask = np.ones_like(chain_feats["res_mask"])
    # if np.sum(flowed_mask) < 1:
    #     raise ValueError("Must be flowed")
    # fixed_mask = 1 - flowed_mask
    # chain_feats["fixed_mask"] = fixed_mask
    # chain_feats["rigids_1"] = gt_bb_rigid.to_tensor_7()
    # chain_feats["sc_ca_t"] = torch.zeros_like(gt_bb_rigid.get_trans())
    #aatype_onehot = F.one_hot(chain_feats["aatype"].clone().detach(), num_classes=args.num_aa_type)
    #chain_feats["aatype_onehot_1"] = aatype_onehot

    #process ec
    ec_class = int(csv_row["ec_class"])
    if args.flow_ec:
        chain_feats["ec_1"] = torch.tensor([ec_class-1]).long()


    # process MSA
    if args.flow_msa:
        reaction_msa_path, enzyme_msa_path = csv_row["processed_reaction_msa_path"], csv_row["processed_enzyme_msa_path"]
        reaction_msa, enzyme_msa = du.read_pkl(reaction_msa_path)["reaction_msa"], du.read_pkl(enzyme_msa_path)["enzyme_msa"]
        msa = torch.cat([enzyme_msa, reaction_msa], dim=-1)[0:args.msa.num_msa]
        
        n_msa, n_token = msa.size()
        if args.msa.num_msa_token <= n_token:
            msa = msa[..., :args.msa.num_msa_token]
        else:
            msa = F.pad(msa, pad=(0, args.msa.num_msa_token-n_token, 0, 0), mode='constant', value=0.)
        
        chain_feats["msa_1"] = msa
        chain_feats["msa_mask"] = msa.clamp(max=1.)

    
    updated_ligand_feats = {}
    # process ligand, move to CoM
    ligand_atom_feat = torch.tensor(ligand_feats["ligand_feat"]).long()
    ligand_atom_coord = torch.tensor(ligand_feats["ligand_pos_after_com"]).double()
    ligand_atom_mask = torch.ones_like(ligand_atom_feat)

    updated_ligand_feats["ligand_atom"] = ligand_atom_feat
    updated_ligand_feats["ligand_pos"] = ligand_atom_coord
    updated_ligand_feats["ligand_mask"] = ligand_atom_mask

    if args.guide_by_condition:
        #process guiding molecule
        product_path = csv_row["processed_product_path"]
        guiding_mol = du.read_pkl(product_path)["product"]

        # guiding_mol = processed_feats["product"]
        guiding_atom_feat = torch.tensor(guiding_mol["molecule_atom_feat"]).long()
        guiding_edge_feat = torch.tensor(guiding_mol["molecule_edge_feat"]).long()
        guiding_edge_index = torch.tensor(guiding_mol["molecule_edge_idx"]).long()
        guiding_atom_mask = torch.ones_like(guiding_atom_feat)
        guiding_edge_mask = torch.ones_like(guiding_edge_feat)
        updated_ligand_feats["guide_ligand_atom"] = guiding_atom_feat
        updated_ligand_feats["guide_ligand_edge"] = guiding_edge_feat
        updated_ligand_feats["guide_ligand_edge_index"] = guiding_edge_index
        updated_ligand_feats["guide_ligand_atom_mask"] = guiding_atom_mask
        updated_ligand_feats["guide_ligand_edge_mask"] = guiding_edge_mask

    #remove unused features
    del chain_feats["residx_atom14_to_atom37"], chain_feats["atom37_pos"], chain_feats["atom37_mask"], chain_feats["atom14_pos"], chain_feats["atom37_pos_before_com"], chain_feats["torsion_angles_sin_cos"]

    return chain_feats, updated_ligand_feats, identifier, ec_class, csv_row


class PdbDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        args,
        gen_model,
        is_training,
    ):
        self._log = logging.getLogger(__name__)
        self.args = args
        self.gen_model = gen_model
        self.is_training = is_training
        self.is_ot = args.ot_plan
        self.ot_reg = args.ot_reg
        self._ot_fn = args.ot_fn.lower()
        self.max_ot_res = args.max_ot_res
        # self.max_mol_len = args.max_mol

        self._init_metadata()

    @property
    def ot_fn(self):
        # import ot as pot
        if self._ot_fn == "exact":
            return pot.emd
        elif self._ot_fn == "sinkhorn":
            return partial(pot.sinkhorn, reg=self.reg)

    def _init_metadata(self):
        """Initialize metadata."""
        pdb_csv = pd.read_csv(self.args.metadata.csv_path)
        if self.args.metadata.filter_num_ligand_atom is not None:
            pdb_csv = pdb_csv[pdb_csv.num_ligand_atom <= self.args.metadata.filter_num_ligand_atom]

        if self.args.metadata.filter_num_protein_atom is not None:
            pdb_csv = pdb_csv[pdb_csv.num_protein_atom <= self.args.metadata.filter_num_protein_atom]

        if self.args.metadata.filter_num_protein_aa is not None:
            pdb_csv = pdb_csv[pdb_csv.num_protein_amino_acid <= self.args.metadata.filter_num_protein_aa]

        if self.args.metadata.subset is not None:
            pdb_csv = pdb_csv[: self.args.metadata.subset]

        pdb_csv = pdb_csv.sort_values("num_ligand_atom", ascending=False)
        self._create_split(pdb_csv)
        

    def _create_split(self, pdb_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = pdb_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            all_ec_classes = np.sort(pdb_csv.ec_class.unique())
            # Fix a random seed to get the same split each time.
            eval_csv = pdb_csv.groupby("ec_class").sample(
                self.args.eval.samples_per_eval_ec, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values("ec_class", ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with ec class {all_ec_classes}"
            )

    def _get_csv_row(self, idx, csv=None):
        if csv is None:
            csv = self.csv

        return get_csv_row(self.args, csv, idx)

    
    def __len__(self):
        return len(self.csv)

    
    def __getitem__(self, idx) -> Any:
        protein, ligand, identifier, ec_class, _ = self._get_csv_row(idx)
        gt_bb_rigid = ru.Rigid.from_tensor_7(protein["rigids_1"])
        gt_trans, gt_rot = extract_trans_rots_mat(gt_bb_rigid)
        protein["trans_1"] = gt_trans
        protein["rot_1"] = gt_rot
        
        # t->1 represents target distribution
        t = np.random.uniform(self.args.min_t, self.args.max_t)
        n_res = protein["aatype"].size(0)
        
        protein["seq_idx"] = torch.arange(n_res) + 1
        aatype_1 = F.one_hot(protein["aatype"], num_classes=self.args.num_aa_type)
        aatype_0 = torch.ones_like(aatype_1) / self.args.num_aa_type
        
        if self.args.flow_msa:
            msa_1 = F.one_hot(protein["msa_1"], num_classes=self.args.msa.num_msa_vocab)
            msa_0 = torch.ones_like(msa_1) / self.args.msa.num_msa_vocab
            n_msa, n_token, _ = msa_1.size()
            

        if self.args.flow_ec: 
            ec_1 = F.one_hot(protein["ec_1"], num_classes=self.args.ec.num_ec_class).reshape(-1)
            ec_0 = torch.ones_like(ec_1) / self.args.ec.num_ec_class
            
            
        if self.is_training and not self.is_ot:
            gen_feats_t = self.gen_model.forward_marginal(
                            rigids_1=gt_bb_rigid, t=t, flow_mask=None, rigids_0=None, center_of_mass=None,
                        )


        elif self.is_training and self.is_ot:
            subset = self.csv[self.csv["ec_class"] == ec_class]
            n_samples = min(subset.shape[0], self.max_ot_res)

            if n_samples == 1 or n_samples == 0:
                # only one sample, we can't do OT
                gen_feats_t = self.gen_model.forward_marginal(
                                rigids_1=gt_bb_rigid, t=t, flow_mask=None, rigids_0=None, center_of_mass=None,
                            )
     

            else:
                sample_subset = subset.sample(
                    n_samples, replace=True, random_state=123
                ).reset_index(drop=True)
                
                list_feat = [
                        self._get_csv_row(i, csv=sample_subset)[0] for i in range(n_samples)
                    ]
    
                list_trans_rot = [
                        extract_trans_rots_mat(
                            ru.Rigid.from_tensor_7(feat["rigids_1"])
                        )
                        for feat in list_feat
                    ]
                list_trans, list_rot = zip(*list_trans_rot)
    
    
                # stack them and change them to torch.tensor
                sample_trans = torch.stack(
                    [torch.from_numpy(trans) for trans in list_trans]
                )
                sample_rot = torch.stack([torch.from_numpy(rot) for rot in list_rot])
    
                rand_rot = torch.tensor(
                    Rotation.random(n_samples * n_res).as_matrix()
                ).to(dtype=sample_rot.dtype)
                rand_rot = rand_rot.reshape(n_samples, n_res, 3, 3)
    
                # random translation
                rand_trans = torch.randn(size=(n_samples, n_res, 3)).to(
                    dtype=sample_trans.dtype
                )
                
                # compute the ground cost for OT: sum of the cost for S0(3) and R3.
                ground_cost = torch.zeros(n_samples, n_samples)
                
                if self.args.aa_ot:
                    # run ot on amino acid
                    list_aatype = [
                        F.one_hot(feat["aatype"], num_classes=self.args.num_aa_type) for feat in list_feat
                    ]
    
                    sample_aatype = torch.stack(
                        [aatype for aatype in list_aatype]
                    )
    
                    rand_aatype = torch.rand(size=(n_samples, n_res, self.args.num_aa_type)).to(
                        dtype=torch.float
                    )
                    rand_aatype = rand_aatype / rand_aatype.sum(dim=-1, keepdim=True)
                    
                    for i in range(n_samples):
                        for j in range(i, n_samples):
                            s03_dist = torch.sum(
                                so3_relative_angle(sample_rot[i], rand_rot[j])
                            )
                            r3_dist = torch.sum(
                                torch.linalg.norm(sample_trans[i] - rand_trans[j], dim=-1)
                            )
                            aa_dist = torch.sum(
                                torch.linalg.norm(sample_aatype[i] - rand_aatype[j], dim=-1)
                            )
                            ground_cost[i, j] = s03_dist**2 + r3_dist**2 + aa_dist**2
                            ground_cost[j, i] = ground_cost[i, j]
    
                else:
                    for i in range(n_samples):
                        for j in range(i, n_samples):
                            s03_dist = torch.sum(
                                so3_relative_angle(sample_rot[i], rand_rot[j])
                            )
                            r3_dist = torch.sum(
                                torch.linalg.norm(sample_trans[i] - rand_trans[j], dim=-1)
                            )
                            ground_cost[i, j] = s03_dist**2 + r3_dist**2
                            ground_cost[j, i] = ground_cost[i, j]

                
                # OT with uniform distributions over the set of pdbs
                a = pot.unif(n_samples, type_as=ground_cost)
                b = pot.unif(n_samples, type_as=ground_cost)
                T = self.ot_fn(a, b, ground_cost)  # NOTE: `ground_cost` is the squared distance on SE(3)^N.
    
                # sample using the plan
                # pick one random indices for the pdb returned by __getitem__
                idx_target = torch.randint(n_samples, (1,))
                pi_target = T[idx_target].squeeze()
                pi_target /= torch.sum(pi_target)
                idx_source = torch.multinomial(pi_target, 1)
                paired_rot = rand_rot[idx_source].squeeze()
                paired_trans = rand_trans[idx_source].squeeze()
                rigids_0 = assemble_rigid_mat(paired_rot, paired_trans)
                gen_feats_t = self.gen_model.forward_marginal(
                    rigids_1=gt_bb_rigid, t=t, flow_mask=None, rigids_0=rigids_0, center_of_mass=None,
                )
    
                if self.args.aa_ot:
                    aatype_0 = rand_aatype[idx_source].squeeze()
    
                            
                # compute the ground cost for msa OT
                if self.args.flow_msa:
                    if self.args.msa_ot:
                        msa_subset = self.csv[self.csv["num_msa"] == n_msa]
                        n_msa_samples = min(subset.shape[0], self.max_ot_res)

                        if n_msa_samples == 1 or n_msa_samples == 0:
                            msa_0 = msa_0

                        else:
                            msa_sample_subset = msa_subset.sample(
                                                n_msa_samples, replace=True, random_state=123
                                            ).reset_index(drop=True)

                            list_msa_feat = [
                                self._get_csv_row(i, csv=msa_sample_subset)[0] for i in range(n_msa_samples)
                            ]
            
                            list_msa = [
                                F.one_hot(feat["msa_1"], num_classes=self.args.msa.num_msa_vocab) for feat in list_msa_feat
                            ]
            
                            sample_msa = torch.stack(
                                [msa for msa in list_msa]
                            )
            
                            rand_msa = torch.rand(size=(n_msa_samples, n_msa, n_token, self.args.msa.num_msa_vocab)).to(
                                dtype=torch.float
                            )
                            rand_msa = rand_msa / rand_msa.sum(dim=-1, keepdim=True)
                            ground_cost_msa = torch.zeros(n_msa_samples, n_msa_samples)
                            for i in range(n_msa_samples):
                                for j in range(i, n_msa_samples):
                                    msa_dist = torch.sum(
                                            torch.linalg.norm(sample_msa[i] - rand_msa[j], dim=-1)
                                        )
                                    
                                    ground_cost_msa[i, j] = msa_dist**2
                                    ground_cost_msa[j, i] = ground_cost_msa[i, j]

                            a = pot.unif(n_msa_samples, type_as=ground_cost_msa)
                            b = pot.unif(n_msa_samples, type_as=ground_cost_msa)
                            T = self.ot_fn(a, b, ground_cost_msa)
                            idx_target = torch.randint(n_msa_samples, (1,))
                            pi_target = T[idx_target].squeeze()
                            pi_target /= torch.sum(pi_target)
                            idx_source = torch.multinomial(pi_target, 1)
                            msa_0 = rand_msa[idx_source].squeeze()
                            
                
        else:
            t = 0.
            gen_feats_t = self.gen_model.sample_ref(
                n_samples=gt_bb_rigid.shape[0],
                flow_mask=None,
                as_tensor_7=True,
                center_of_mass=None,
            )

        
        # sample vectorfields for amino acid and msa
        if self.is_training:
            if self.args.discrete_flow_type == 'uniform':
                aatype_t = self.gen_model.forward_multinomial(
                    feat_0=aatype_0,
                    feat_1=protein["aatype"],
                    t=t,
                    flow_mask=None,
                )

            elif self.args.discrete_flow_type == 'masking':
                aatype_0 = torch.rand(n_res)
                aatype_t = self.gen_model.forward_masking(
                    feat_0=aatype_0,
                    feat_1=protein["aatype"],
                    t=t,
                    mask_token_idx=self.args.masked_aa_token_idx,
                    flow_mask=None,
                )


            if self.args.flow_msa:
                if self.args.discrete_flow_type == 'uniform':
                    msa_t = self.gen_model.forward_multinomial(
                        feat_0=msa_0,
                        feat_1=protein["msa_1"],
                        t=t,
                        flow_mask=None,
                    )
    
                elif self.args.discrete_flow_type == 'masking':
                    msa_0 = torch.rand(n_msa, n_token)
                    msa_t = self.gen_model.forward_masking(
                        feat_0=msa_0,
                        feat_1=protein["msa_1"],
                        t=t,
                        mask_token_idx=self.args.msa.masked_msa_token_idx,
                        flow_mask=None,
                    ).reshape(n_msa, n_token)


            
            if self.args.flow_ec:
                if self.args.discrete_flow_type == 'uniform':
                    ec_t = self.gen_model.forward_multinomial(
                        feat_0=ec_0,
                        feat_1=protein["ec_1"],
                        t=t,
                        flow_mask=None,
                    )
    
                elif self.args.discrete_flow_type == 'masking':
                    ec_0 = torch.rand(1).reshape(-1)
                    ec_t = self.gen_model.forward_masking(
                        feat_0=ec_0,
                        feat_1=protein["ec_1"],
                        t=t,
                        mask_token_idx=self.args.ec.masked_ec_token_idx,
                        flow_mask=None,
                    )
                
          
        else:
            if self.args.discrete_flow_type == 'uniform':
                aatype_t = self.gen_model.forward_multinomial(
                    feat_0=aatype_0,
                    feat_1=None,
                    t=0.,
                    flow_mask=None,
                )

                if self.args.flow_msa:
                    msa_t = self.gen_model.forward_multinomial(
                        feat_0=msa_0,
                        feat_1=None,
                        t=0.,
                        flow_mask=None,
                    )
    
                if self.args.flow_ec:
                    ec_t = self.gen_model.forward_multinomial(
                        feat_0=ec_0,
                        feat_1=None,
                        t=0.,
                        flow_mask=None,
                    )

            elif self.args.discrete_flow_type == 'masking':
                aatype_0 = torch.rand(n_res)
                aatype_t = self.gen_model.forward_masking(
                    feat_0=aatype_0,
                    feat_1=None,
                    t=0.,
                    mask_token_idx=self.args.masked_aa_token_idx,
                    flow_mask=None,
                )
            
                if self.args.flow_msa:
                    msa_0 = torch.rand(n_msa, n_token)
                    msa_t = self.gen_model.forward_masking(
                        feat_0=msa_0,
                        feat_1=None,
                        t=0.,
                        mask_token_idx=self.args.msa.masked_msa_token_idx,
                        flow_mask=None,
                    ).reshape(n_msa, n_token)
    
                if self.args.flow_ec:
                    ec_0 = torch.rand(1).reshape(-1)
                    ec_t = self.gen_model.forward_masking(
                        feat_0=ec_0,
                        feat_1=None,
                        t=0.,
                        mask_token_idx=self.args.ec.masked_ec_token_idx,
                        flow_mask=None,
                    )


        protein["aatype_t"] = aatype_t

        if self.args.flow_msa:
            protein["msa_t"] = msa_t
            

        if self.args.flow_ec:
            protein["ec_t"] = ec_t

        protein.update(gen_feats_t)
        protein["t"] = t

        final_feats = {}
        for k, v in protein.items():
            if not torch.is_tensor(v):
                v = torch.tensor(v)
            
            if k in {"residx_atom14_to_atom37", "atom37_pos", "atom14_pos", "atom37_mask"}:
                continue

            else:
                final_feats[k] = v

        final_feats.update(ligand)

        if self.is_training:
            return final_feats
            
        else:
            return final_feats, identifier

            