import math
import torch
from torch import nn

from flowmatch.data.utils import calc_distogram

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


class EdgeFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        #   c_s, c_z, relpos_k, template_type):
        super(EdgeFeatureNet, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.embed.c_s
        self.c_z = self._cfg.embed.c_z
        self.feat_dim = self._cfg.embed.feat_dim

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)
        self.linear_relpos = nn.Linear(self.feat_dim, self.feat_dim)

        total_edge_feats = self.feat_dim * 3 + self._cfg.embed.num_bins * 2 + 2

        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.ReLU(),
            nn.Linear(self.c_z, self.c_z),
            nn.LayerNorm(self.c_z),
        )

    def embed_relpos(self, r):
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self.feat_dim, max_len=2056)
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, edge_mask, flow_mask):
        # Input: [b, n_res, c_s]
        num_batch, num_res, _ = s.shape

        # [b, n_res, c_z]
        p_i = self.linear_s_p(s)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # [b, n_res]
        r = torch.arange(
            num_res, device=s.device).unsqueeze(0).repeat(num_batch, 1)
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.embed.num_bins)
        sc_feats = calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.embed.num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]

        diff_feat = self._cross_concat(flow_mask[..., None], num_batch, num_res)
        all_edge_feats.append(diff_feat)
        
        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1).to(torch.float))
        edge_feats *= edge_mask.unsqueeze(-1)
        return edge_feats