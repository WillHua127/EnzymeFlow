import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP



class MolMPNN(nn.Module):
    def __init__(self, model_conf):
        super(MolMPNN, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf

        self.node_embed_dims = self._model_conf.mpnn.mpnn_node_embed_size
        self.edge_embed_dims = self._model_conf.mpnn.mpnn_edge_embed_size

        self.node_embedder = nn.Sequential(
            nn.Embedding(self._model_conf.num_atom_type, self.node_embed_dims),
            nn.SiLU(),
            nn.Linear(self.node_embed_dims, self.node_embed_dims),
            nn.LayerNorm(self.node_embed_dims),
        )

        # self.node_outter = nn.Sequential(
        #     nn.Linear(self.node_embed_dims, self.node_embed_dims),
        #     nn.SiLU(),
        #     nn.Linear(self.node_embed_dims, self.node_embed_dims),
        #     nn.LayerNorm(self.node_embed_dims),
        # )

        self.edge_embedder = nn.Sequential(
            nn.Embedding(self._model_conf.mpnn.num_edge_type, self.edge_embed_dims),
            nn.SiLU(),
            nn.Linear(self.edge_embed_dims, self.edge_embed_dims),
            nn.LayerNorm(self.edge_embed_dims),
        )
        
        self.mpnn = AttentiveFP(
                                in_channels=self.node_embed_dims,
                                hidden_channels=self.node_embed_dims,
                                out_channels=self.node_embed_dims,
                                edge_dim=self.edge_embed_dims,
                                num_layers=self._model_conf.mpnn.mpnn_layers,
                                num_timesteps=self._model_conf.mpnn.n_timesteps,
                                dropout=self._model_conf.mpnn.dropout,
                            )

    def dense_to_sparse(
        self,
        mol_atom,
        mol_edge,
        mol_edge_feat,
        mol_atom_mask,
        mol_edge_mask,
    ):
        mol_atom_list = mol_atom[mol_atom_mask]
        mol_edge_feat_list = mol_edge_feat[mol_edge_mask]

        if mol_edge.size(dim=1) == 2:
            mol_edge = mol_edge.transpose(1,2)
        mol_edge_list = [edge[mask] for edge, mask in zip(mol_edge, mol_edge_mask)]
        
        n_nodes = mol_atom_mask.sum(dim=1, keepdim=True)
        cum_n_nodes = torch.cumsum(n_nodes, dim=0)
        new_mol_edge_list = [mol_edge_list[0]]
        for edge, size in zip(mol_edge_list[1:], cum_n_nodes[:-1]):
            new_mol_edge = edge + size
            new_mol_edge_list.append(new_mol_edge)
            
        new_mol_edge_list = torch.cat(new_mol_edge_list, dim=0)

        if new_mol_edge_list.size(dim=1) == 2:
            new_mol_edge_list = new_mol_edge_list.transpose(1,0)

        idx = 0
        batch_mask = []
        for size in n_nodes:
            batch_mask.append(torch.zeros(size, dtype=torch.long) + idx)
            idx += 1
        batch_mask = torch.cat(batch_mask).to(mol_atom.device)
        
        return mol_atom_list, new_mol_edge_list, mol_edge_feat_list, batch_mask
        
        


    def forward(
        self,
        mol_atom,
        mol_edge,
        mol_edge_feat,
        mol_atom_mask,
        mol_edge_mask,
    ):
        n_batch = mol_atom.size(0)
        
        mol_atom_mask = mol_atom_mask.bool()
        mol_edge_mask = mol_edge_mask.bool()
        mol_atom, mol_edge, mol_edge_feat, batch_mask = self.dense_to_sparse(mol_atom, mol_edge, mol_edge_feat, mol_atom_mask, mol_edge_mask)
        assert mol_edge.size(1) == mol_edge_feat.size(0)

        mol_atom = self.node_embedder(mol_atom)
        mol_edge_feat = self.edge_embedder(mol_edge_feat)
        
        mol_rep = self.mpnn(mol_atom, mol_edge, mol_edge_feat, batch_mask)
        #mol_rep = self.node_outter(mol_rep)

        return mol_rep