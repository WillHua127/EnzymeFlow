""" Vectorfield network module.
The structure of this file is greatly influenced by SE3 Diffusion by Yim et. al 2023
Link: https://github.com/jasonkyuyim/se3_diffusion
"""
import functools as fn
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from ofold.utils.rigid_utils import Rigid

from model import ipa_pytorch, pairformer
from flowmatch.data import all_atom
from flowmatch.data import utils as du
from model import msaformer, molmpnn, node_feature_net, edge_feature_net


def outer_sum(x, y):
    return x[..., None, :] + y[..., None, :, :]

def produce_pair_repr(m, n, padding=False, method='repeat'):
    # (bs,n,d), (bs,m,d) -> (bs, n+m, n+m, d)
    assert method in ['repeat', 'outer_sum']
    
    bs = m.size(0)
    assert n.size(0) == bs
    m_size = m.size(1)
    n_size = n.size(1)
    n_node = m_size + n_size
    
    if padding: 
        n_node = n_node + 1

    if method == 'repeat':
        # (bs, m, d) -> (bs, m+n, d)
        m = F.pad(m, pad=(0, 0, 0, n_node-m_size), value=0) 
        
        # (bs, m+n, d) -> (bs, m+n, m+n, d) 
        m = m.unsqueeze(1).repeat(1, n_node, 1, 1)
        n = n.unsqueeze(1).repeat(1, n_size, 1, 1)
        
    elif method == 'outer_sum':
        # (bs, m, d) -> (bs, m, m, d)
        m = outer_sum(m, m)
        n = outer_sum(n, n)

        # (bs, n, n, d) -> (bs, n+m, n+m, d)
        m = F.pad(m, pad=(0, 0, 0, n_node-m_size, 0, n_node-m_size), value=0)
        
    mask = torch.zeros(bs, n_node, n_node).to(m.device)
    mask[:, :m_size, :m_size] = 1.

    if padding:
        mask[:, m_size+1:, m_size+1:] = 1.
        m[:, m_size+1:, m_size+1:, :] = n
    else:
        mask[:, m_size:, m_size:] = 1.
        m[:, m_size:, m_size:, :] = n

    m = m * mask.reshape(bs, n_node, n_node, 1)
    return m



def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


class Embedder(nn.Module):
    def __init__(self, model_conf):
        super(Embedder, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        # Time step embedding
        index_embed_size = self._embed_conf.index_embed_size
        t_embed_size = index_embed_size
        node_embed_dims = t_embed_size + 1
        edge_in = (t_embed_size + 1) * 2

        # Sequence index embedding
        node_embed_dims += index_embed_size
        edge_in += index_embed_size

        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Linear(node_embed_dims, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        if self._embed_conf.embed_self_conditioning:
            edge_in += self._embed_conf.num_bins
        edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(edge_in, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        self.timestep_embedder = fn.partial(
            get_timestep_embedding, embedding_dim=self._embed_conf.index_embed_size
        )
        self.index_embedder = fn.partial(
            get_index_embedding, embed_size=self._embed_conf.index_embed_size
        )

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res**2, -1])
        )

    def forward(
        self,
        *,
        seq_idx,
        t,
        flow_mask,
        self_conditioning_ca,
    ):
        
        num_batch, num_res = seq_idx.shape
        node_feats = []

        # Set time step to epsilon=1e-5 for fixed residues.
        prot_t_embed = torch.tile(
            self.timestep_embedder(t)[:, None, :], (1, num_res, 1)
        )
        prot_t_embed = torch.cat([prot_t_embed, flow_mask[..., None]], dim=-1)
        node_feats = [prot_t_embed]
        pair_feats = [self._cross_concat(prot_t_embed, num_batch, num_res)]

        # Positional index features.
        node_feats.append(self.index_embedder(seq_idx))
        rel_seq_offset = seq_idx[:, :, None] - seq_idx[:, None, :]
        rel_seq_offset = rel_seq_offset.reshape([num_batch, num_res**2])
        pair_feats.append(self.index_embedder(rel_seq_offset))

        # Self-conditioning distogram.
        if self._embed_conf.embed_self_conditioning:
            sc_dgram = du.calc_distogram(
                self_conditioning_ca,
                self._embed_conf.min_bin,
                self._embed_conf.max_bin,
                self._embed_conf.num_bins,
            )
            pair_feats.append(sc_dgram.reshape([num_batch, num_res**2, -1]))

        node_embed = self.node_embedder(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embedder(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([num_batch, num_res, num_res, -1])
        return node_embed, edge_embed


class MSAEmbedder(nn.Module):
    def __init__(self, model_conf):
        super(MSAEmbedder, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf
        self._msa_conf = model_conf.msa

        self.msa_encoder = msaformer.MSATransformer(
                            vocab_size=self._msa_conf.num_msa_vocab, 
                            n_layers=self._msa_conf.msa_layers,
                            n_heads=self._msa_conf.msa_heads,
                            model_depth=self._msa_conf.msa_embed_size,
                            ff_depth=self._msa_conf.msa_hidden_size, 
                            dropout=self._model_conf.dropout,
                        )

        self.col_attn = msaformer.MultiHeadAttention(
                num_heads=self._msa_conf.msa_heads, 
                embed_dim=self._msa_conf.msa_embed_size,
            )

        self.row_attn = msaformer.MultiHeadAttention(
                num_heads=self._msa_conf.msa_heads, 
                embed_dim=self._msa_conf.msa_embed_size,
            )

    def forward(
        self,
        msa_feature,
        msa_mask=None,
    ):
        bs, n_msa, n_token = msa_feature.size()
        msa_feature = msa_feature.reshape(bs*n_msa, n_token)
        msa_embed = self.msa_encoder(msa_feature).reshape(bs, n_msa, n_token, -1)
        msa_embed = msa_embed.transpose(1, 2).reshape(bs*n_token, n_msa, -1)

        if msa_mask is not None:
            msa_mask = msa_mask.transpose(1, 2).reshape(bs*n_token, n_msa)
        msa_embed = self.col_attn(msa_embed, msa_embed, mask=msa_mask).reshape(bs, n_token, n_msa, -1).transpose(1, 2)
        msa_embed = msa_embed.reshape(bs*n_msa, n_token, -1)

        if msa_mask is not None:
            msa_mask = msa_mask.reshape(bs, n_token, n_msa)
            msa_mask = msa_mask.transpose(1, 2).reshape(bs*n_msa, n_token)
        msa_embed = self.row_attn(msa_embed, msa_embed, mask=msa_mask).reshape(bs, n_msa, n_token, -1)
        return msa_embed
        


class MolEmbedder(nn.Module):
    def __init__(self, model_conf):
        super(MolEmbedder, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        node_embed_dims = self._model_conf.num_atom_type
        node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Embedding(node_embed_dims, node_embed_size, padding_idx=0),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        self.node_aggregator = nn.Sequential(
            nn.Linear(node_embed_size + self._model_conf.edge_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.LayerNorm(node_embed_size),
        )

        self.dist_min = self._model_conf.ligand_rbf_d_min
        self.dist_max = self._model_conf.ligand_rbf_d_max
        self.num_rbf_size = self._model_conf.num_rbf_size
        self.edge_embed_size = self._model_conf.edge_embed_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(self.num_rbf_size + node_embed_size + node_embed_size, self.edge_embed_size),
            nn.ReLU(),
            nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
            nn.ReLU(),
            nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
            nn.LayerNorm(self._model_conf.edge_embed_size),
        )

        mu = torch.linspace(self.dist_min, self.dist_max, self.num_rbf_size)
        self.mu = mu.reshape([1, 1, 1, -1])
        self.sigma = (self.dist_max - self.dist_min) / self.num_rbf_size

    def coord2dist(self, coord, edge_mask):
        n_batch, n_atom = coord.size(0), coord.size(1)
        radial = torch.sum((coord.unsqueeze(1) - coord.unsqueeze(2)) ** 2, dim=-1)
        dist = torch.sqrt(
                radial + 1e-10
            ) * edge_mask

        radial = radial * edge_mask
        return radial, dist
    
    def rbf(self, dist):
        dist_expand = torch.unsqueeze(dist, -1)
        _mu = self.mu.to(dist.device)
        rbf = torch.exp(-(((dist_expand - _mu) / self.sigma) ** 2))
        return rbf

    def forward(
        self,
        ligand_atom,
        ligand_pos,
        edge_mask,
    ):
        num_batch, num_atom = ligand_atom.shape
        node_embed = self.node_embedder(ligand_atom)
        
        radial, dist = self.coord2dist(
                            coord=ligand_pos, 
                            edge_mask=edge_mask,
                        )


        edge_embed = self.rbf(dist) * edge_mask[..., None]
        src_node_embed = node_embed.unsqueeze(1).repeat(1, num_atom, 1, 1)
        tar_node_embed = node_embed.unsqueeze(2).repeat(1, 1, num_atom, 1)
        edge_embed = torch.cat([src_node_embed, tar_node_embed, edge_embed], dim=-1)
        edge_embed = self.edge_embedder(edge_embed.to(torch.float))

        src_node_agg = (edge_embed.sum(dim=1) / (edge_mask[..., None].sum(dim=1)+1e-10)) * ligand_atom.clamp(max=1.)[..., None]
        src_node_agg = torch.cat([node_embed, src_node_agg], dim=-1)
        node_embed = node_embed + self.node_aggregator(src_node_agg)

        return node_embed, edge_embed


class DistEmbedder(nn.Module):
    def __init__(self, model_conf):
        super(DistEmbedder, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf
        self._embed_conf = model_conf.embed

        edge_embed_size = self._model_conf.edge_embed_size

        self.dist_min = self._model_conf.bb_ligand_rbf_d_min
        self.dist_max = self._model_conf.bb_ligand_rbf_d_max
        self.num_rbf_size = self._model_conf.num_rbf_size
        self.edge_embedder = nn.Sequential(
            nn.Linear(self.num_rbf_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.LayerNorm(edge_embed_size),
        )

        mu = torch.linspace(self.dist_min, self.dist_max, self.num_rbf_size)
        self.mu = mu.reshape([1, 1, 1, -1])
        self.sigma = (self.dist_max - self.dist_min) / self.num_rbf_size

    def coord2dist(self, coord, edge_mask):
        n_batch, n_atom = coord.size(0), coord.size(1)
        radial = torch.sum((coord.unsqueeze(1) - coord.unsqueeze(2)) ** 2, dim=-1)
        dist = torch.sqrt(
                radial + 1e-10
            ) * edge_mask

        radial = radial * edge_mask
        return radial, dist
    
    def rbf(self, dist):
        dist_expand = torch.unsqueeze(dist, -1)
        _mu = self.mu.to(dist.device)
        rbf = torch.exp(-(((dist_expand - _mu) / self.sigma) ** 2))
        return rbf

    def forward(
        self,
        rigid,
        ligand_pos,
        bb_ligand_mask,
    ):
        curr_bb_pos = all_atom.to_atom37(Rigid.from_tensor_7(torch.clone(rigid)))[-1][:, :, 1].to(ligand_pos.device)

        curr_bb_lig_pos = torch.cat([curr_bb_pos, ligand_pos], dim=1)
        edge_mask = bb_ligand_mask.unsqueeze(dim=1) * bb_ligand_mask.unsqueeze(dim=2)
        
        radial, dist = self.coord2dist(
                            coord=curr_bb_lig_pos, 
                            edge_mask=edge_mask,
                        )


        edge_embed = self.rbf(dist) * edge_mask[..., None]
        edge_embed = self.edge_embedder(edge_embed.to(torch.float))

        return edge_embed
        


class ProteinLigandNetwork(nn.Module):
    def __init__(self, model_conf):
        super(ProteinLigandNetwork, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf

        # self.embedding_layer = Embedder(model_conf)
        self.node_feature_net = node_feature_net.NodeFeatureNet(model_conf)
        self.edge_feature_net = edge_feature_net.EdgeFeatureNet(model_conf)
        
        self.mol_embedding_layer = MolEmbedder(model_conf)
        self.ipanet = ipa_pytorch.IpaNetwork(model_conf)
        # self.pairformer = pairformer.PairformerStack(model_conf)


        #node fusion
        self.node_embed_size = self._model_conf.node_embed_size
        self.node_embedder = nn.Sequential(
            nn.Embedding(self._model_conf.num_aa_type, self.node_embed_size),
            nn.ReLU(),
            nn.Linear(self.node_embed_size, self.node_embed_size),
            nn.LayerNorm(self.node_embed_size),
        )

        self.node_pair_embedder = nn.Sequential(
            nn.Linear(self.node_embed_size, self.node_embed_size),
            nn.ReLU(),
            nn.Linear(self.node_embed_size, self.node_embed_size),
            nn.LayerNorm(self.node_embed_size),
        )

        self.node_fusion = nn.Sequential(
            nn.Linear(self.node_embed_size + self.node_embed_size, self.node_embed_size),
            nn.ReLU(),
            nn.Linear(self.node_embed_size, self.node_embed_size),
            nn.LayerNorm(self.node_embed_size),
        )

        self.bb_lig_fusion = msaformer.CrossAttention(
                query_input_dim=self.node_embed_size,
                key_input_dim=self.node_embed_size,
                output_dim=self.node_embed_size,
            )

        #edge fusion
        self.edge_embed_size = self._model_conf.edge_embed_size
        self.edge_dist_embedder = DistEmbedder(model_conf)
        self.edge_fusion = nn.Sequential(
            nn.Linear(model_conf.pairformer.dim_single + model_conf.edge_embed_size, self.edge_embed_size),
            nn.ReLU(),
            nn.Linear(self._model_conf.edge_embed_size, self._model_conf.edge_embed_size),
            nn.LayerNorm(self._model_conf.edge_embed_size),
        )

        self.aatype_pred_net = nn.Sequential(
                nn.Linear(self.node_embed_size, self.node_embed_size),
                nn.ReLU(),
                nn.Linear(self.node_embed_size, self.node_embed_size),
                nn.ReLU(),
                nn.Linear(self.node_embed_size, model_conf.num_aa_type),
                # nn.Softmax(dim=-1),
            )


        self.pretrain = self._model_conf.pretrain_kd_pred
        if self.pretrain:
            self.lig_bb_fusion = msaformer.CrossAttention(
                query_input_dim=self.node_embed_size,
                key_input_dim=self.node_embed_size,
                output_dim=self.node_embed_size,
            )
            
            self.kd_pred = nn.Sequential(
                nn.Linear(self.node_embed_size, self.node_embed_size),
                nn.ReLU(),
                nn.Linear(self.node_embed_size, self.node_embed_size),
                nn.ReLU(),
                nn.Linear(self.node_embed_size, 1),
            )
        
            

    def _apply_mask(self, aatype_diff, aatype_0, diff_mask):
        return diff_mask * aatype_diff + (1 - diff_mask) * aatype_0

    def forward(self, input_feats, use_context=False):
        """Forward computes the reverse conditionals p(X^t|X^{t+1})
        for each item in the batch

        Args:
            X: the noised samples from the noising process, of shape [Batch, N, D].
                Where the T time steps are t=1,...,T (i.e. not including the un-noised X^0)

        Returns:
            model_out: dictionary of model outputs.
        """
        # Frames as [batch, res, 7] tensors.
        bb_mask = input_feats["res_mask"].type(torch.float32)  # [B, N]
        flow_mask = input_feats["flow_mask"].type(torch.float32)
        edge_mask = bb_mask[..., None] * bb_mask[..., None, :]

        n_batch, n_res = bb_mask.shape

        # Initial embeddings of positonal and relative indices.
        # init_bb_node_embed, init_bb_edge_embed = self.embedding_layer(
        #     seq_idx=input_feats["seq_idx"],
        #     t=input_feats["t"],
        #     flow_mask=flow_mask,
        #     self_conditioning_ca=input_feats["sc_ca_t"],
        # )

        init_bb_node_embed = self.node_feature_net(
            t=input_feats["t"],
            res_mask=bb_mask,
            flow_mask=flow_mask,
            pos=input_feats["seq_idx"],
            aatypes=input_feats["aatype_t"],
            aatypes_sc=input_feats["sc_aa_t"],
        )

        init_bb_edge_embed = self.edge_feature_net(
            s=init_bb_node_embed,
            t=input_feats["trans_t"],
            sc_t=input_feats["sc_ca_t"],
            edge_mask=edge_mask,
            flow_mask=flow_mask,
        )
        
        bb_node_embed = init_bb_node_embed * bb_mask[..., None]
        bb_edge_embed = init_bb_edge_embed * edge_mask[..., None]

        
        # Amino-Acid embedding
        bb_aa_embed = self.node_embedder(input_feats["aatype_t"]) * bb_mask[..., None]
        bb_aa_embed = torch.cat([bb_aa_embed, bb_node_embed], dim=-1)
        bb_node_embed = self.node_fusion(bb_aa_embed)
        bb_node_embed = bb_node_embed * bb_mask[..., None]
        

        lig_mask = input_feats["ligand_mask"]
        lig_edge_mask = lig_mask[..., None] * lig_mask[..., None, :]
        # Initial embeddings of ligands.
        lig_init_node_embed, _ = self.mol_embedding_layer(
                ligand_atom=input_feats["ligand_atom"],
                ligand_pos=input_feats["ligand_pos"],
                edge_mask=lig_edge_mask,
            )
        lig_node_embed = lig_init_node_embed * lig_mask[..., None]


        bb_lig_rep, _ = self.bb_lig_fusion(
                                query_input=bb_node_embed, 
                                key_input=lig_node_embed, 
                                value_input=lig_node_embed, 
                                query_input_mask=bb_mask, 
                                key_input_mask=lig_mask,
                            )


        # node fusion
        bb_node_embed = bb_node_embed + bb_lig_rep

        
        bb_ligand_mask = torch.cat([bb_mask, lig_mask], dim=-1)
        bb_lig_edge = self.edge_dist_embedder(
            rigid=input_feats["rigids_t"],
            ligand_pos=input_feats["ligand_pos"],
            bb_ligand_mask=bb_ligand_mask,
        )
        bb_lig_edge = bb_lig_edge[:, :n_res, :n_res, :] * edge_mask[..., None]

        # edge fusion
        bb_edge_embed = bb_edge_embed + bb_lig_edge
        
        # Run main network
        model_out = self.ipanet(bb_node_embed, bb_edge_embed, input_feats)

        # Node embed 
        node_embed = model_out["node_embed"] * bb_mask[..., None]

        if self.pretrain:
            _, n_lig = lig_mask.shape

            lig_rep, _ = self.lig_bb_fusion(
                                query_input=lig_node_embed, 
                                key_input=node_embed, 
                                value_input=node_embed, 
                                query_input_mask=lig_mask, 
                                key_input_mask=bb_mask,
                            )
            
            # kd_pred = self.kd_pred(torch.cat([node_embed, lig_rep], dim=1))
            # kd_pred = kd_pred.sum(dim=1) / (bb_mask.sum(dim=1, keepdim=True) + lig_mask.sum(dim=1, keepdim=True) + 1e-10)
            kd_pred = self.kd_pred(lig_rep)
            kd_pred = kd_pred.sum(dim=1) /  (lig_mask.sum(dim=1, keepdim=True) + 1e-10)
            kd_pred = kd_pred.reshape(-1)
            

        # Amino Acid prediction
        aa_pred = self.aatype_pred_net(node_embed) * bb_mask[..., None]
        
        # Psi angle prediction
        # gt_psi = input_feats["torsion_angles_sin_cos"][..., 2, :]
        # psi_pred = self._apply_mask(model_out["psi"], gt_psi, 1 - fixed_mask[..., None])

        pred_out = {
            "amino_acid": aa_pred,
            "rigids_tensor": model_out["rigids"],
            # "psi": psi_pred,
        }
        
        if self.pretrain:
            pred_out["kd"] = kd_pred

        pred_out["rigids"] = model_out["rigids"].to_tensor_7()
        return pred_out
