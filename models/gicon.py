# Copyright (c) Institute of Information Processing, Leibniz University Hannover. All Rights Reserved.

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from util import box_ops
from util.misc import (NestedTensor)
from .backbone import build_backbone
from .transformer import GraphEncoder, ImageEncoder

class GICon(nn.Module):
    """ Simple and Stable Scene Graph and Image Contrastive Learning """
    def __init__(self, backbone, node_bbox=False, image_layer_num=6, graph_layer_num=6):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py

        """
        super().__init__()
        self.node_bbox = node_bbox

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.image_encoder = ImageEncoder(backbone=backbone, d_model=512, nhead=8, nlayer=image_layer_num,
                                          d_ffn=1024, dropout=0.1, activation="relu")
        self.graph_encoder = GraphEncoder(node_bbox=node_bbox, d_model=512, nhead=8, nlayer=graph_layer_num,
                                          d_ffn=1024, dropout=0.1, activation="relu")


    def forward(self, samples: NestedTensor, graphs):
        graph_embeddings, graph_mask, graph_pos = self.graph_encoder(graphs)

        image_embeddings, patch_mask, patch_pos = self. image_encoder(samples)

        graph_features = graph_embeddings[:, 0]
        image_features = image_embeddings[:, 0]
        graph_features = graph_features / graph_features.norm(dim=1, keepdim=True)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        entry = {}
        entry['logits_per_image'] = logit_scale * image_features @ graph_features.t()
        entry['logits_per_graph'] = entry['logits_per_image'].t()
        return entry

class SetCriterion(nn.Module):
    """ This class computes the loss for RelTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def loss_contrastive(self, outputs, targets, log=True):
        gt = torch.arange(len(outputs['logits_per_image']), device=outputs['logits_per_image'].device).long()
        loss = F.cross_entropy(outputs['logits_per_image'], gt) + F.cross_entropy(outputs['logits_per_graph'], gt)
        losses = {'loss_cont': loss/2}
        return losses


    def get_loss(self, loss, outputs, targets, **kwargs):
        loss_map = {'contrastive': self.loss_contrastive
            }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets))

        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class GraphEvaluator:

    def __init__(self):

        self.counter = 0
        self.image_to_graph_matched = 0
        self.graph_to_image_matched = 0

    def eval_image_graph_matching(self, targets, entry):

        ground_truth_align = torch.range(0, entry['logits_per_image'].shape[0]-1,
                                         device=entry['logits_per_image'].device)
        self.counter += ground_truth_align.shape[0]

        logits_image_to_graph = F.softmax(entry['logits_per_image'], dim=1)
        logits_graph_to_image = F.softmax(entry['logits_per_graph'], dim=1)

        image_to_graph_matching = sum(logits_image_to_graph.argmax(dim=1)==ground_truth_align).item()
        graph_to_image_matching = sum(logits_graph_to_image.argmax(dim=1)==ground_truth_align).item()

        self.image_to_graph_matched += image_to_graph_matching
        self.graph_to_image_matched += graph_to_image_matching


    def print_result(self):
        r_precision_image_to_graph_matching = self.image_to_graph_matched/self.counter
        r_precision_graph_to_image_matching = self.graph_to_image_matched/self.counter

        print('='*10)
        print('R Precision for image-graph matching: ', r_precision_image_to_graph_matching)
        print('R Precision for graph-image matching: ', r_precision_graph_to_image_matching)


def build(args):

    device = torch.device(args.device)

    backbone = build_backbone(args)

    model = GICon(backbone, node_bbox=args.node_bbox, image_layer_num=args.image_layer_num,
                  graph_layer_num=args.graph_layer_num)

    weight_dict = {'loss_cont': 1}
    losses = ['contrastive']

    criterion = SetCriterion(weight_dict=weight_dict, losses=losses)
    criterion.to(device)

    evaluator = GraphEvaluator()

    return model, criterion, evaluator

