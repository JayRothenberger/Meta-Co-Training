# standard library imports
import argparse
from copy import deepcopy as copy
import pickle
import time
import os
# installed imports
import torch
import torchvision
import wandb
# local code imports
from DAHS.DAHB import DistributedAsynchronousGridSearch
from utils import subset_npercent_dataset
from DAHS.torch_utils import sync_parameters, setup, cleanup

from MCT import MetaCoTrainingModel

from image_distances import IMAGE_DISTANCES, IMAGE_TRANSFORMS
from utils import LinearProbe, FPFT, FinetunedLinearProbe


def training_process(args, rank, world_size):
    dict_args = vars(args)
    device = int(os.environ['RANK']) % torch.cuda.device_count()

    assert args.view in ['DINOv2', 'CLIP']

    if args.view == 'DINOv2':
        args.batch_size = min(args.batch_size, 64)

    views = [args.view]
    trains = []
    unlbls = []
    vals = []
    # TODO: fix this for when the gpus are not evenly divisible
    if rank < len(views):
        wandb.init(project=f'MCT cold start {args.dataset} 1', entity='ai2es',
        name=f"({args.view}) {rank}: {args.train_size}",
        config={'args': dict_args})

    for view in views:
        dataset = torchvision.datasets.ImageNet(args.dataset_path, split='train', transform=IMAGE_TRANSFORMS[view])
        val = torchvision.datasets.ImageNet(args.dataset_path, split='val', transform=IMAGE_TRANSFORMS[view])

        train, unlbl = subset_npercent_dataset(dataset, percent=args.train_size * 100)
        trains.append(train)
        unlbls.append(unlbl)
        vals.append(val)

    num_classes = 1000

    models = [IMAGE_DISTANCES[view]().model.to(device) for view in views]

    for model in models:
        model.to(device)
    
    shapes = []
    with torch.no_grad():
        for train, model, view in zip(trains, models, views):
            print(os.environ['RANK'], view, device)
            shapes.append(model(train[0][0].unsqueeze(0).to(device)).shape[-1])

    probes = [LinearProbe(model, shape, num_classes, args.train_temp, args.mlp) for shape, model in zip(shapes, models)]

    MCTModel = MetaCoTrainingModel(probes)

    # fine-tuning stage in which the model does not alter embedder weights
    states = MCTModel.train(args.epochs, args.epochs + 1, copy(trains), copy(unlbls), copy(vals), checkpoint_path=f'./chkpts/{args.view}_chkpt', batch_size=args.batch_size, log_interval=100)

    # synchronize those bad boys
    MCTModel.reduce_weights()

    models = [FPFT(model, args.fpft_temp) for model in probes]
    
    MCTModel = MetaCoTrainingModel(models)
    torch.distributed.barrier()

    # full-parameter finetuning
    states = MCTModel.train(10, 10, copy(trains), copy(unlbls), copy(vals), checkpoint_path=f'./chkpts/{args.view}_chkpt_fpft', batch_size=args.batch_size, log_interval=100, approx=False)

        # synchronize those bad boys
    MCTModel.reduce_weights()

    models = [FinetunedLinearProbe(model) for model in models]

    if int(os.environ["RANK"]) == 0:
        for i, (m, a) in enumerate(zip(models, MCTModel.best_val_accs)):
            with open(f'./prepped_views/{args.view}_{round(float(a.cpu()), 4)}', 'wb') as fp:
                pickle.dump(models[i], fp)

    return states


def main(args, rank, world_size):
    setup(rank, world_size)

    search_space = ['train_size', 'view', 'fpft_temp', 'train_temp', ]

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
    
    parser.add_argument('-e', '--epochs', type=int, default=128, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=128, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-p', '--patience', type=int, default=32, 
                        help='patience for training')
    parser.add_argument('-fpfpt', '--fpft_temp', type=float, default=[0.125, 1, 8], help='temperature for head fpft')
    parser.add_argument('-tt', '--train_temp', type=float, default=[0.125, 1, 8], help='temperature for head training')
    parser.add_argument('-m', '--mlp', type=bool, default=[True, False], help='temperature for head training')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=384, 
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('--view', type=str, default= ['DINOv2', 'CLIP'], help='view to prepare')
    parser.add_argument('--hparam_path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/one_percent_cold_start', help='path for hparam search directory')
    parser.add_argument('--dataset_path', type=str, default='/ourdisk/hpc/ai2es/datasets/Imagenet/2012', help='path containing training dataset')
    parser.add_argument('--train_size', type=float, default=[0.01], help='size of the training set (%)')
    parser.add_argument('--balanced', type=bool, default=False, 
                        help='Balanced dataset subsetting if true, else stratified sampling')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    main(args, rank, world_size)
