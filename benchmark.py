# Copyright (c) Institute of Information Processing, Leibniz University Hannover.

import argparse
import random
import pickle
import numpy as np
import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import datasets.transforms as T
from datasets.coco import ConvertCocoPolysToMask
import util.misc as utils
from engine import evaluate
from models import build_model

class ModelPredictionDataset(Dataset):
    def __init__(self, args):
        '''
        :param args:
        label index from 1 -> 151 classes with dummy label
        relation index from 1 -> 51 classes with dummy label
        bbox -> int, xywh
        '''
        self.img_folder = args.img_folder

        with open(args.prediction, 'rb') as f:
            self.predictions = pickle.load(f)

        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._transforms = T.Compose([T.Resize(size=512),
                                      normalize])
        self.prepare = ConvertCocoPolysToMask(False)


    def __getitem__(self, idx):
        prediction = self.predictions[idx]
        image_id = prediction['image_id']
        img = Image.open(os.path.join(self.img_folder, '{}.jpg'.format(image_id))).convert("RGB")
        target = []
        for i in range(len(prediction['labels'])):
            tmp = {}
            tmp['image_id'] = image_id
            tmp['segmentation'] = None
            tmp['area'] = 0
            tmp['iscrowd'] = 0
            tmp['id'] = 0
            tmp['bbox'] = prediction['boxes'][i].astype(int).tolist()
            tmp['category_id'] = prediction['labels'][i]
            target.append(tmp)

        rel_target = prediction['rel_annotations']
        np.random.shuffle(rel_target)
        sampled_entities = []
        sampled_triplets= []
        i = 0
        while len(sampled_entities)<10 and len(sampled_triplets)<len(rel_target):
            b1, b2, _ = rel_target[i]
            if len(np.unique(sampled_entities+[b1,b2])) <= 10:
                if b1 not in sampled_entities:
                    sampled_entities.append(b1)
                if b2 not in sampled_entities:
                    sampled_entities.append(b2)
                sampled_triplets.append(rel_target[i])
                i += 1
            else:
                break

        np.random.shuffle(sampled_entities)
        sampled_entities = list(sampled_entities)
        reindex_triplets = []
        for triplet in sampled_triplets:
            reindex_triplets.append([sampled_entities.index(triplet[0]),
                                     sampled_entities.index(triplet[1]),
                                     triplet[2]])
        sampled_triplets = reindex_triplets

        target = {'image_id': image_id, 'annotations': target, 'rel_annotations': sampled_triplets}

        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # reorder boxes, labels
        target['boxes'] = target['boxes'][sampled_entities]
        target['labels'] = target['labels'][sampled_entities]

        if target['boxes'].shape[0] < 10:
            target['boxes'] = torch.cat([target['boxes'], torch.zeros([10-target['boxes'].shape[0], 4])], dim=0) # padding to 10
            target['labels'] = torch.cat([target['labels'], 151*torch.ones(10-target['labels'].shape[0], dtype=torch.int64)], dim=0)# padding to 10, class index 151
        return img, target

    def __len__(self):
        return len(self.predictions)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=75, type=int)
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--image_layer_num', default=6, type=int,
                        help="Number of encoding layers in the image transformer")
    parser.add_argument('--graph_layer_num', default=6, type=int,
                        help="Number of encoding layers in the graph transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # Setting
    parser.add_argument('--node_bbox', action='store_true',
                        help="Enable node bounding boxes for input graph")

    # dataset parameters
    parser.add_argument('--dataset', default='vg')
    parser.add_argument('--img_folder', default='data/vg/images/', type=str,
                        help="image data folder")
    parser.add_argument('--prediction', type=str, help="the prediction results of scene graph generation methods")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser


def main(args):
    utils.init_distributed_mode(args)
    torch.multiprocessing.set_sharing_strategy('file_system')
    print("git:\n  {}\n".format(utils.get_sha()))
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, evaluator = build_model(args)
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_val = ModelPredictionDataset(args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=True, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    evaluate(model, criterion, evaluator, data_loader_val, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
