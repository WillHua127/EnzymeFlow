import math
import torch
from torch import nn

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
    


class NodeFeatureNet(nn.Module):
    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.embed.c_s
        self.c_pos_emb = self._cfg.embed.c_pos_emb
        self.c_timestep_emb = self._cfg.embed.c_timestep_emb
        embed_size = self.c_pos_emb + self.c_timestep_emb * 2 + 1

        self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
        embed_size += self.c_s + self.c_timestep_emb + self._cfg.num_aa_type
            
        self.linear = nn.Sequential(
            nn.Linear(embed_size, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.ReLU(),
            nn.Linear(self.c_s, self.c_s),
            nn.LayerNorm(self.c_s),
        )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_timestep_embedding(
            timesteps,
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            t,
            res_mask,
            flow_mask,
            pos,
            aatypes,
            aatypes_sc,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            flow_mask[..., None],
            self.embed_t(t, res_mask),
            self.embed_t(t, res_mask)
        ]
        input_feats.append(self.aatype_embedding(aatypes))
        input_feats.append(self.embed_t(t, res_mask))
        input_feats.append(aatypes_sc)
        return self.linear(torch.cat(input_feats, dim=-1))