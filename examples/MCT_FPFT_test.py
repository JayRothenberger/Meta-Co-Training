import numpy as np
import torch
import torchvision
import argparse
import os
from copy import deepcopy as copy

from torchvision import datasets

import pickle
import time
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from math import floor
import wandb


def training_process(args, rank, world_size):
    dict_args = vars(args)

    from models import FCNN
    from utils import subset_npercent_dataset
    from MCT import MetaCoTrainingModel
    from torchvision.transforms import v2
    from image_distances import IMAGE_DISTANCES, IMAGE_TRANSFORMS

    torch.manual_seed(13)

    # train0, unlbl0, val0, num_classes = make_concat_dataset(args.view0, 'EsViT', dataset=args.dataset, percent=args.train_size * 100, balanced=args.balanced)
    # train1, unlbl1, val1, num_classes = make_concat_dataset(args.view1, 'MAE', dataset=args.dataset, percent=args.train_size * 100, balanced=args.balanced)
    # TODO: 
    # function that splits the data using the ImageDataset for the appropriate challenge
    # build the models that we need to finetune
    # 
    # views = ['DINOv2', 'CLIP', 'EsViT']
    views = ['DINOv2', 'CLIP', 'SigLIP']
    trains = []
    unlbls = []
    vals = []
    # TODO: fix this for when the gpus are not evenly divisible
    if rank < len(views):
        wandb.init(project=f'MCT test {args.dataset}', entity='ai2es',
        name=f"{rank}: {args.train_size}",
        config={'args': vars(args)})

    for view in views:
        dataset = torchvision.datasets.ImageNet('/ourdisk/hpc/ai2es/datasets/Imagenet/2012', split='train', transform=IMAGE_TRANSFORMS[view])
        val = torchvision.datasets.ImageNet('/ourdisk/hpc/ai2es/datasets/Imagenet/2012', split='val', transform=IMAGE_TRANSFORMS[view])

        train, unlbl = subset_npercent_dataset(dataset, percent=args.train_size * 100)
        trains.append(train)
        unlbls.append(unlbl)
        vals.append(val)

    print('device count', torch.cuda.device_count())

    num_classes = 1000

    models = [IMAGE_DISTANCES[view]().model for view in views]

    for model in models:
        model.to(0)
    
    with torch.no_grad():
        shapes = [model(train[0][0].unsqueeze(0).to(0)).shape[-1] for train, model in zip(trains, models)]

    probes = [LinearProbe(model, shape, num_classes) for shape, model in zip(shapes, models)]

    # MCTModel = MetaCoTrainingModel(probes)

    # fine-tuning stage in which the model does not alter embedder weights

    trains1, unlbls1, vals1 = copy(trains), copy(unlbls), copy(vals)

    # states = MCTModel.train(1, 2, trains, unlbls, vals, batch_size=args.batch_size, log_interval=100)

    # synchronize those bad boys
    # MCTModel.reduce_weights()
    # pickle them
    # start = time.time()
    # if int(os.environ["RANK"]) == 0:
    #     for i, m in enumerate(MCTModel.models):
    #         with open(f'models_list_{i}', 'wb') as fp:
    #             pickle.dump(MCTModel.models[i], fp)
    # else:
    #     time.sleep(60)

    model_list = []
    
    for i, m in enumerate(models):
        rankpath = f'pretrainbf16-384-x05_{i}'

        #while os.path.getctime(rankpath) < start:
        #    time.sleep(1)

        try:
            with open(rankpath, 'rb') as fp:
                model_list.append(pickle.load(fp))
        except Exception as e:
            print(os.environ["RANK"], ':', e)

    models = [FPFT(model) for model in model_list]

    models = [models[0], models[-1]]
    trains1 = [trains1[0], trains1[-1]]
    unlbls1 = [unlbls1[0], unlbls1[-1]]
    vals1 = [vals1[0], vals1[-1]]
    
    MCTModel = MetaCoTrainingModel(models)
    torch.distributed.barrier()
    states = MCTModel.train(10, 10, copy(trains1), copy(unlbls1), copy(vals1), checkpoint_path='no_fpft_mct', batch_size=args.batch_size, log_interval=100, approx=False)

    models = [FinetunedLinearProbe(model) for model in models]


    MCTModel = MetaCoTrainingModel(models)
    torch.distributed.barrier()
    states = MCTModel.train(args.epochs, 10, copy(trains1), copy(unlbls1), copy(vals1), checkpoint_path='no_fpft_mct_after', batch_size=args.batch_size, log_interval=100, approx=False)


    return states

def setup(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

def main(args, rank, world_size):
    setup(rank, world_size)

    from DAHS.DAHB import DistributedAsynchronousGridSearch
    from DAHS.torch_utils import sync_parameters

    search_space = ['train_size']

    agent = sync_parameters(args, rank, search_space, DistributedAsynchronousGridSearch)

    args = agent.to_namespace(agent.combination)

    states = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)

    print('cleanup')
    cleanup()


        
def create_parser():
    parser = argparse.ArgumentParser(description='MCT benchmark')
    
    parser.add_argument('-e', '--epochs', type=int, default=1280, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-p', '--patience', type=int, default=32, 
                        help='patience for training')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=64, 
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('--dataset', type=str, default='IN1k', metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/one_percent_man',
                        help='path for hparam search directory')
    parser.add_argument('--train_size', type=float, default=[0.01],
                        help='size of the training set (%)')
    parser.add_argument('--balanced', type=bool, default=False, 
                        help='Balanced dataset subsetting if true, else stratified sampling')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method('spawn')

    main(args, rank, world_size)
