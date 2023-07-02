# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from util.misc import (NestedTensor, nested_tensor_from_tensor_list)

class GraphEncoder(nn.Module):

    def __init__(self, node_bbox=False, d_model=512, nhead=8, nlayer=6, d_ffn=2048,
                 dropout=0.1, activation="relu"):
        super().__init__()
        self.node_bbox = node_bbox
        if self.node_bbox:
            self.box_embedd = nn.Sequential(nn.Linear(4, 128),
                                            nn.BatchNorm1d(128),
                                            nn.Linear(128, d_model))
            self.norm = nn.LayerNorm(d_model)

        self.graph_cls = nn.Parameter(torch.zeros(1, d_model))
        self.entity_embed = nn.Embedding(152, d_model)
        self.relation_embed = nn.Embedding(52, d_model)

        self.node_encodings = nn.Parameter(torch.zeros(10, d_model))

        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, activation)
        self.layers = _get_clones(encoder_layer, nlayer)
        self.nlayer = nlayer

        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graphs):
        nodes = torch.stack([self.entity_embed(g["labels"]) for g in graphs])
        if self.node_bbox:
            node_boxes = torch.stack([self.box_embedd(g["boxes"]) for g in graphs])
            nodes = self.norm(nodes+node_boxes)
        nodes_encodings = self.node_encodings.unsqueeze(0).repeat(len(graphs), 1, 1)
        nodes_mask = torch.stack([g["labels"] == 151 for g in graphs])

        edges = torch.stack([self.relation_embed(g["rel_annotations"][:,2]) for g in graphs])
        edges_encodings = torch.stack([self.node_encodings[g["rel_annotations"][:,0]] -
                                       self.node_encodings[g["rel_annotations"][:,1]] for g in graphs])
        edges_mask = torch.stack([g["rel_annotations"][:,2] == 51 for g in graphs])

        graph_cls = self.graph_cls.unsqueeze(0).repeat(len(graphs), 1, 1)
        graph_mask = torch.zeros([len(graphs), 1], dtype=torch.bool, device=graphs[0]["labels"].device)

        output = torch.cat([graph_cls, nodes, edges], dim=1)
        pos = torch.cat([torch.zeros([len(graphs), 1, self.d_model], device=graphs[0]["labels"].device), nodes_encodings, edges_encodings], dim=1)
        mask = torch.cat([graph_mask, nodes_mask, edges_mask], dim=1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=mask, pos=pos)

        return output, mask, pos

class ImageEncoder(nn.Module):

    def __init__(self, backbone=None, d_model=512, nhead=8, nlayer=6, d_ffn=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, d_model, kernel_size=1)
        self.image_cls = nn.Parameter(torch.zeros(1, d_model))
        torch.nn.init.xavier_uniform_(self.image_cls)

        encoder_layer = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout, activation)
        self.layers = _get_clones(encoder_layer, nlayer)

        self.nlayer = nlayer
        self.d_model = d_model
        self.nhead = nhead

    def forward(self, samples: NestedTensor):

        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        output = self.input_proj(src).flatten(start_dim=2).transpose(1,2)
        mask = mask.flatten(start_dim=1)
        pos = pos[-1].flatten(start_dim=2).transpose(1,2)

        image_cls = self.image_cls.unsqueeze(0).repeat(output.shape[0], 1, 1)
        image_mask = torch.zeros([output.shape[0], 1], dtype=torch.bool, device=output.device)

        output = torch.cat([image_cls, output], dim=1)
        pos = torch.cat([torch.zeros([output.shape[0], 1, self.d_model], device=output.device), pos], dim=1)
        mask = torch.cat([image_mask, mask], dim=1)

        for layer in self.layers:
            output = layer(output, src_key_padding_mask=mask, pos=pos)

        return output, mask, pos

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_key_padding_mask: Optional[Tensor] = None, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



class TransformerDecoderLayer(nn.Module):
    """decoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.activation = _get_activation_fn(activation)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, graph_embeddings, graph_mask, graph_pos,
                image_embeddings, patch_mask, patch_pos):

        tgt2 = self.cross_attn(query=self.with_pos_embed(graph_embeddings, graph_pos),
                               key=self.with_pos_embed(image_embeddings, patch_pos),
                               value=image_embeddings,
                               key_padding_mask=patch_mask)[0]
        tgt = graph_embeddings + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
