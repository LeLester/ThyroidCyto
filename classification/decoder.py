from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.modules.transformer import _get_activation_fn




class TransformerDecoderLayerOptimal(nn.Module):
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5) -> None:
        super(TransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)


        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt




class Decoder(nn.Module):

    def __init__(self, num_classes, num_token,  decoder_embedding=1536,
                 initial_num_features=1536):
        super(Decoder, self).__init__()
        embed_len_decoder = num_token
        self.norm = nn.LayerNorm(decoder_embedding)
        # non-learnable queries
        query_embed_1 = nn.Embedding(num_classes, decoder_embedding)
        query_embed_1.requires_grad_(False)

        query_embed_2 = nn.Embedding(num_classes, decoder_embedding//2)
        query_embed_2.requires_grad_(False)
        #
        query_embed_3 = nn.Embedding(num_classes, decoder_embedding//4)
        query_embed_3.requires_grad_(False)

        # decoder
        decoder_dropout = 0.1
        num_layers_decoder = 1
        dim_feedforward = 4096


        layer_decode_1 = TransformerDecoderLayerOptimal(d_model=decoder_embedding, nhead=8, dim_feedforward=dim_feedforward, dropout=decoder_dropout)
        self.decoder_1 = nn.TransformerDecoder(layer_decode_1, num_layers=num_layers_decoder)
        self.decoder_1.query_embed = query_embed_1
        self.fc1 = nn.Linear(decoder_embedding, decoder_embedding//2)

        layer_decode_2 = TransformerDecoderLayerOptimal(d_model=decoder_embedding//2, nhead=4, dim_feedforward=dim_feedforward//2, dropout=decoder_dropout)
        self.decoder_2 = nn.TransformerDecoder(layer_decode_2, num_layers=num_layers_decoder)
        self.decoder_2.query_embed = query_embed_2
        self.fc2 = nn.Linear(decoder_embedding//2, decoder_embedding//4)

        layer_decode_3 = TransformerDecoderLayerOptimal(d_model=decoder_embedding//4, nhead=2, dim_feedforward=dim_feedforward//4, dropout=decoder_dropout)
        self.decoder_3 = nn.TransformerDecoder(layer_decode_3, num_layers=num_layers_decoder)
        self.decoder_3.query_embed = query_embed_3
        self.fc3 = nn.Linear(decoder_embedding//4, 1)

        self.norm1 = nn.LayerNorm(decoder_embedding//2)
        self.norm2 = nn.LayerNorm(decoder_embedding//4)


        self.q1_q2 = nn.Linear(decoder_embedding, decoder_embedding//2)
        self.norm_q2 = nn.LayerNorm(decoder_embedding//2)

        self.q2_q3 = nn.Linear(decoder_embedding//2, decoder_embedding//4)
        self.norm_q3 = nn.LayerNorm(decoder_embedding//4)

    def forward(self, x):
        x = self.norm(x)
        # x = torch.nn.functional.relu(x, inplace=True)
        x = torch.nn.functional.relu(x)
        bs = x.shape[0]
        x = x.transpose(0, 1)

        query_embed_1 = self.decoder_1.query_embed.weight
        q_1 = query_embed_1.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        x_1 = self.decoder_1(q_1, x)
        x_1 = self.fc1(x_1)
        x_1 = self.norm1(x_1)
        # x_1 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = torch.nn.functional.relu(x_1)

        query_embed_2 = self.decoder_2.query_embed.weight
        q_2 = query_embed_2.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        x_2 = x_1 + self.decoder_2(q_2, x_1)
        x_2 = self.fc2(x_2)
        x_2 = self.norm2(x_2)
        # x_2 = torch.nn.functional.relu(x_2, inplace=True)
        x_2 = torch.nn.functional.relu(x_2)

        query_embed_3 = self.decoder_3.query_embed.weight
        q_3 = query_embed_3.unsqueeze(1).expand(-1, bs, -1)  # no allocation of memory with expand
        x_3 = x_2 + self.decoder_3(q_3, x_2)
        x_3 = self.fc3(x_3)


        x_3 = x_3.transpose(0, 1).flatten(1)

        logits = x_3
        return logits
