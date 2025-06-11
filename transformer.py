import torch
import torch.nn as nn
import numpy as np
import sys

class GraphTransformerVals(nn.Module):
    def __init__(self, num_categories, num_index, embedding_dim, num_heads, num_layers, dropout):
        super(GraphTransformerVals, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.index_step_pre_embedding = nn.Embedding(num_index, embedding_dim)
        self.index_start_pre_embedding = nn.Embedding(num_index, embedding_dim)
        nn.init.zeros_(self.index_start_pre_embedding.weight)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embedding_dim, 1)  # FIXME Do we need a hidden layer?

    def forward(self, src_val, src_cat, idx_step, idx_start):
        mask = src_val.isnan()
        src_val2 = torch.where(mask, 0, src_val)
        src = src_val2[:,:,None] * self.embedding(src_cat)

        src += self.index_step_pre_embedding(idx_step).unsqueeze(1)
        src += self.index_start_pre_embedding(idx_start).unsqueeze(1)

        transformer_output = self.transformer(src, src_key_padding_mask=mask)
        output = self.fc(transformer_output)
        return output



