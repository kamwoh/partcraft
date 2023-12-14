from typing import Optional

import torch
from torch import nn


class TokenMapper(nn.Module):
    def __init__(self,
                 num_parts,
                 num_k_per_part,
                 out_dims,
                 projection_nlayers=1,
                 projection_activation=nn.ReLU(),
                 with_pe=True):
        super().__init__()

        self.num_parts = num_parts
        self.num_k_per_part = num_k_per_part
        self.with_pe = with_pe
        self.out_dims = out_dims

        self.embedding = nn.Embedding((self.num_k_per_part + 1) * num_parts, out_dims)
        if with_pe:
            self.pe = nn.Parameter(torch.randn(num_parts, out_dims))
        else:
            self.register_buffer('pe', torch.zeros(num_parts, out_dims))

        if projection_nlayers == 0:
            self.projection = nn.Identity()
        else:
            projections = []
            for i in range(projection_nlayers - 1):
                projections.append(nn.Linear(out_dims, out_dims))
                projections.append(projection_activation)

            projections.append(nn.Linear(out_dims, out_dims))
            self.projection = nn.Sequential(*projections)

    def get_all_embeddings(self, no_projection=False, no_pe=False):
        idx = torch.arange(self.num_parts * (self.num_k_per_part + 1)).long().to(self.embedding.weight.device)
        idx = idx.reshape(self.num_parts, self.num_k_per_part + 1)
        emb = self.embedding(idx)  # (K, N, d)

        if not no_pe:
            emb_pe = emb + self.pe.unsqueeze(1)
        else:
            emb_pe = emb

        if not no_projection:
            projected = self.projection(emb_pe)
        else:
            projected = emb_pe

        return projected

    def forward(self, hashes, index: Optional[torch.Tensor] = None):
        B = hashes.size(0)

        # 0, 257, 514, ...
        if index is None:
            offset = torch.arange(self.num_parts, device=hashes.device) * (self.num_k_per_part + 1)
            hashes = self.embedding(hashes.long() + offset.reshape(1, -1))  # (B, N, d)
        else:
            offset = index.reshape(-1) * (self.num_k_per_part + 1)
            hashes = self.embedding(hashes.long() + offset.reshape(B, -1).long())  # (B, N, d)

        if index is not None:
            pe = self.pe[index.reshape(-1)]  # index must be equal size
            pe = pe.reshape(B, -1, self.out_dims)
            hashes = hashes + pe
        else:
            hashes = hashes + self.pe.unsqueeze(0).repeat(B, 1, 1)
        projected = self.projection(hashes)

        return projected
