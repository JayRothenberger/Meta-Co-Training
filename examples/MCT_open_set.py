# standard library imports
import argparse
from copy import deepcopy as copy
import os
import shutil
import pickle
import numpy as np
# installed imports
import torch
import torchvision
import wandb
from tqdm import tqdm
# local code imports
from mct.dahps import DistributedAsynchronousRandomSearch, sync_parameters

from mct.MCT import MetaCoTrainingModel
from mct.image_models import IMAGE_DISTANCES, IMAGE_TRANSFORMS
from mct.models import FPFT, FinetunedLinearProbe, MLPProbe, FCNN
from mct.utils import subset_npercent_dataset, IndexedDataset


def training_process(args, rank, world_size):
    dict_args = vars(args)
    device = int(os.environ['RANK']) % torch.cuda.device_count()


    torch.manual_seed(13)
    torch.cuda.set_device(device)

    views = ['DINOv2', 'CLIP']

    view = views[int(os.environ['RANK']) % len(views)]

    trains = []
    unlbls = []
    vals = []

    # each view gets its own weights and biases process to monitor resources and performance
    if rank < len(views):
        wandb.init(project=f'MCT e2e with unc {args.dataset}', entity='ai2es',
        name=f"{rank}: {args.train_size}",
        config={'args': dict_args})
                
    shapes = []

    for view in views:
        # get the path 
        trainpath = f'/ourdisk/hpc/ai2es/jroth/statistical distances/{view}_IN1K_train.ds'
        valpath = f'/ourdisk/hpc/ai2es/jroth/statistical distances/{view}_IN1K_val.ds'
        shardpath = f'/ourdisk/hpc/ai2es/jroth/Meta-Co-Training/data_processing/LAION-embed-{view}/'

        with open(trainpath, 'rb') as fp:
            x_train, y_train = pickle.load(fp)

        with open(valpath, 'rb') as fp:
            x_val, y_val = pickle.load(fp)

        shards = os.listdir(shardpath)

        x_unlbls, y_unlbls = [], []

        x_unlbl = torch.cat(x_unlbls)
        y_unlbl = torch.cat(y_unlbls)

        train = torch.utils.data.TensorDataset(torch.tensor(x_train).type(torch.float16), torch.tensor(y_train))
        val = torch.utils.data.TensorDataset(torch.tensor(x_val).type(torch.float16), torch.tensor(y_val))

        if args.train_size < 1.0:
            train, unlbl = subset_npercent_dataset(train, percent=args.train_size * 100)
        else:
            for shard in tqdm(sorted(shards)[:25]):
                with open(os.path.join(shardpath, shard), 'rb') as fp:
                    x_unlbl, y_unlbl = pickle.load(fp)
                    x_unlbls.append(torch.tensor(x_unlbl).type(torch.float16))
                    y_unlbls.append(torch.tensor(y_unlbl))

            unlbl = IndexedDataset(torch.utils.data.TensorDataset(x_unlbl, y_unlbl))

        trains.append(train)
        unlbls.append(unlbl)
        vals.append(val)
        shapes.append(train[0][0].shape[-1])
        assert train[0][0].shape[-1] == unlbl[0][1].shape[-1] == val[0][0].shape[-1], (view, train[0][0].shape[-1], unlbl[0][0].shape[-1], val[0][0].shape[-1])

    num_classes = 1000

    models = [FCNN(shape, num_classes, [1024, 1024, 1024]) for shape in shapes]

    MCTModel = MetaCoTrainingModel(models, accum_steps=1)

    # preparation stage in which the model does not alter embedder weights
    print('warmup')
    states = MCTModel.train(args.warmup_epochs, args.warmup_epochs + 1, copy(trains), copy(unlbls), copy(vals), copy(vals), checkpoint_path=f'./chkpts/{view}_chkpt', batch_size=args.batch_size, log_interval=100, amp=args.amp)

    MCTModel = MetaCoTrainingModel(models, accum_steps=1)
    torch.distributed.barrier()
    print('MCT')
    states = MCTModel.train(args.epochs, 0, copy(trains), copy(unlbls), copy(vals), copy(vals), checkpoint_path='no_fpft_mct_after', batch_size=args.batch_size, log_interval=100, approx=args.approx, amp=args.amp)

    return states


def setup(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def main(args, rank, world_size):
    setup(rank, world_size)
    search_space = ['train_size', 'batch_size', 'learning_rate', 'approx', 'amp']

    agent = sync_parameters(args, rank, search_space, DistributedAsynchronousRandomSearch)

    args = agent.to_namespace(agent.combination)

    states = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)

    print('cleanup')
    cleanup()

        
def create_parser():
    parser = argparse.ArgumentParser(description='MCT benchmark')
    
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                        help='warmup epochs (default: 10)')
    parser.add_argument('--fpft_epochs', type=int, default=15, 
                        help='fpft epochs (default: 10)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=[4096, 8192, 16384], 
                        help='batch size for training (default: 64)')
    parser.add_argument('-p', '--patience', type=int, default=32, 
                        help='patience for training')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=1024,
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=[1e-3, 1e-4],
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('--dataset', type=str, default='IN1k', metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/open_set_hp2', help='path for hparam search directory')
    parser.add_argument('--dataset_path', type=str, default='/ourdisk/hpc/ai2es/datasets/Imagenet/2012', help='path containing training dataset')
    parser.add_argument('--train_size', type=float, default=[1.0],
                        help='size of the training set (%)')
    parser.add_argument('--balanced', type=bool, default=False, 
                        help='Balanced dataset subsetting if true, else stratified sampling')
    parser.add_argument('--approx', type=bool, default=[True, False], 
                        help='Perform the full-parameter finetuning step')
    parser.add_argument('--amp', type=bool, default=[True, False], 
                        help='Perform the full-parameter finetuning step')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method('spawn')

    main(args, rank, world_size)
