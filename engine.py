# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import torch
import util.misc as utils

@torch.no_grad()
def evaluate(model, criterion, evaluator, data_loader, device, args):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 100, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = pre_processing(targets)

        outputs = model(samples, targets)

        evaluator.eval_image_graph_matching(targets, outputs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.print_result()
    print("Averaged stats:", metric_logger)

def pre_processing(targets):
    edge_max_num = max([len(t['rel_annotations']) for t in targets])
    for t in targets:
        if t['rel_annotations'].shape[0] < edge_max_num:
            t['rel_annotations'] = torch.cat([t['rel_annotations'],
                                              torch.tensor([[0, 0, 51]],
                                               dtype=torch.long,
                                               device=t['rel_annotations'].device).repeat(
                                                edge_max_num - t['rel_annotations'].shape[0], 1)], dim=0)

    return targets