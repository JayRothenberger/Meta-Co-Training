# standard library imports
import argparse
from copy import deepcopy as copy
import os
# installed imports
import torch
import torchvision
import wandb
# local code imports
from mct.dahps import DistributedAsynchronousGridSearch, sync_parameters

from mct.MCT import MetaCoTrainingModel
from mct.image_models import IMAGE_DISTANCES, IMAGE_TRANSFORMS
from mct.models import FPFT, FinetunedLinearProbe, MLPProbe
from mct.utils import subset_npercent_dataset



def training_process(args, rank, world_size):
    dict_args = vars(args)
    device = int(os.environ['RANK']) % torch.cuda.device_count()


    torch.manual_seed(13)
    torch.cuda.set_device(device)

    views = ['DINOv2', 'CLIP', 'SigLIP', 'EsViT']

    view = views[int(os.environ['RANK']) % len(views)]

    trains = []
    unlbls = []
    vals = []

    # each view gets its own weights and biases process to monitor resources and performance
    if rank < len(views):
        wandb.init(project=f'MCT e2e multiview {args.dataset}', entity='ai2es',
        name=f"{rank}: {args.train_size}",
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

    models = [MLPProbe(model, shape, num_classes) for shape, model in zip(shapes, models)]

    MCTModel = MetaCoTrainingModel(models)

    # preparation stage in which the model does not alter embedder weights
    states = MCTModel.train(args.warmup_epochs, args.warmup_epochs + 1, copy(trains), copy(unlbls), copy(vals), copy(vals), checkpoint_path=f'./chkpts/{view}_chkpt', batch_size=args.batch_size, log_interval=100, amp=True)

    if args.fpft:
        # full-parameter fine-tuning on their respective datasets (this is exclusively supervised)
        models = [FPFT(model) for model in models]
        
        MCTModel = MetaCoTrainingModel(models)
        torch.distributed.barrier()
        states = MCTModel.train(args.fpft_epochs, args.fpft_epochs + 1, copy(trains), copy(unlbls), copy(vals), copy(vals), checkpoint_path='no_fpft_mct', batch_size=args.batch_size, log_interval=100, approx=False, amp=True)

        # now the meta co-training step with the representation frozen which empirically prevents collapse
        models = [FinetunedLinearProbe(model) for model in models]

    MCTModel = MetaCoTrainingModel(models)
    torch.distributed.barrier()
    states = MCTModel.train(args.epochs, 0, copy(trains), copy(unlbls), copy(vals), copy(vals), checkpoint_path='no_fpft_mct_after', batch_size=args.batch_size, log_interval=100, approx=False, amp=True)

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
    
    parser.add_argument('--warmup_epochs', type=int, default=45, 
                        help='warmup epochs (default: 10)')
    parser.add_argument('--fpft_epochs', type=int, default=45, 
                        help='fpft epochs (default: 10)')
    parser.add_argument('--epochs', type=int, default=150, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=512, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-p', '--patience', type=int, default=32, 
                        help='patience for training')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=64, 
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('--dataset', type=str, default='IN1k', metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/one_percent_cold_start', help='path for hparam search directory')
    parser.add_argument('--dataset_path', type=str, default='/ourdisk/hpc/ai2es/datasets/Imagenet/2012', help='path containing training dataset')
    parser.add_argument('--train_size', type=float, default=[0.01],
                        help='size of the training set (%)')
    parser.add_argument('--balanced', type=bool, default=False, 
                        help='Balanced dataset subsetting if true, else stratified sampling')
    parser.add_argument('--fpft', type=bool, default=False, 
                        help='Perform the full-parameter finetuning step')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.multiprocessing.set_start_method('spawn')

    main(args, rank, world_size)
