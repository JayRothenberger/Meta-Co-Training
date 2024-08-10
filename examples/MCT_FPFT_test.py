# standard library imports
import argparse
from copy import deepcopy as copy
import pickle
import os
# installed imports
import torch
import torchvision
import wandb
# local code imports
from DAHS.DAHB import DistributedAsynchronousGridSearch
from DAHS.torch_utils import sync_parameters

from MCT import MetaCoTrainingModel

from mct.image_models import IMAGE_DISTANCES, IMAGE_TRANSFORMS
from mct.models import FPFT, FinetunedLinearProbe
from mct.utils import subset_npercent_dataset



def training_process(args, rank, world_size):
    dict_args = vars(args)

    torch.manual_seed(13)

    views = ['DINOv2', 'CLIP']
    trains = []
    unlbls = []
    vals = []

    # each view gets its own weights and biases process to monitor resources and performance
    if rank < len(views):
        wandb.init(project=f'MCT test {args.dataset}', entity='ai2es',
        name=f"{rank}: {args.train_size}",
        config={'args': dict_args})

    for view in views:
        dataset = torchvision.datasets.ImageNet(args.dataset_path, split='train', transform=IMAGE_TRANSFORMS[view])
        val = torchvision.datasets.ImageNet(args.dataset_path, split='val', transform=IMAGE_TRANSFORMS[view])
        # this function will know if we should use the index subset files rather than a random subset
        train, unlbl = subset_npercent_dataset(dataset, percent=args.train_size * 100)
        trains.append(train)
        unlbls.append(unlbl)
        vals.append(val)

    print('device count', torch.cuda.device_count())

    models = [IMAGE_DISTANCES[view]().model for view in views]

    for model in models:
        model.to(0)
    
    trains1, unlbls1, vals1 = copy(trains), copy(unlbls), copy(vals)

    model_list = []
    
    for i, m in enumerate(models):
        rankpath = f'pretrainbf16-384-x05_{i}'

        try:
            with open(rankpath, 'rb') as fp:
                model_list.append(pickle.load(fp))
        except Exception as e:
            print(os.environ["RANK"], ':', e)

    # full-parameter fine-tuning on their respective datasets (this is exclusively supervised)
    models = [FPFT(model) for model in model_list]

    models = [models[0], models[-1]]
    trains1 = [trains1[0], trains1[-1]]
    unlbls1 = [unlbls1[0], unlbls1[-1]]
    vals1 = [vals1[0], vals1[-1]]
    
    MCTModel = MetaCoTrainingModel(models)
    torch.distributed.barrier()
    states = MCTModel.train(10, 10, copy(trains1), copy(unlbls1), copy(vals1), checkpoint_path='no_fpft_mct', batch_size=args.batch_size, log_interval=100, approx=False)

    # now the meta co-training step with the representation frozen which empirically prevents collapse
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
    parser.add_argument('--hparam_path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/one_percent_cold_start', help='path for hparam search directory')
    parser.add_argument('--dataset_path', type=str, default='/ourdisk/hpc/ai2es/datasets/Imagenet/2012', help='path containing training dataset')
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
