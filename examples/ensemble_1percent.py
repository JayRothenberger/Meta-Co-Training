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
from mct.utils import subset_npercent_dataset, IndexedDataset, ConcatDataset
from mct.train import model_run
from mct.MCT import make_loader

from abc import ABC, abstractmethod
import torch


class Ensemble(torch.nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = members

    def forward(self, x, with_variance=False):
        predictions = []
        for member in self.members:
            predictions.append(member(x))

        predictions = torch.stack(predictions, 0)

        if not self.train:
          if with_variance:
              # return the prediction with the uncertainty value
              pred = (torch.sum(predictions, 0) / predictions.shape[0]) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
              pred_idx = torch.argmax(pred, -1)
              smax = torch.nn.functional.softmax(predictions, -1)
              unc = 1.0 - torch.gather(smax, -1, pred_idx.unsqueeze(-1)).squeeze()
              return pred, unc
          else:
              # return the calibrated prediction
              return (torch.sum(predictions, 0) / len(self.members)) / (1 + (0.3*torch.var(predictions, 0)))**(0.5)
        else:
          return torch.sum(predictions, 0) / len(self.members)


class Metric(ABC):
    """
    abstract metric class
    """
    def __init__(self):
        self.value = None

    @abstractmethod
    def step(self, output, target):
        pass

    @abstractmethod
    def accumulate(self):
        pass

class Accuracy(Metric):
    def __init__(self):
        self.value = None
        self.sum = 0.0
        self.total = 0.0
        self.fn = lambda output, target: (torch.argmax(output, -1) == target).type(torch.float32).mean()

    def step(self, output, target):
        self.sum += self.fn(output, target)
        self.total += 1

    def accumulate(self):
        if self.total == 0.0:
            return None
        self.value = (self.sum / self.total).cpu().item()



@torch.no_grad()
def accuracy_callback(model, loader, name='train'):
        metric = Accuracy()
        model.eval()

        for batch in loader:
            x, y = batch
            
            out = model(x.to(torch.cuda.current_device()))

            metric.step(out.cpu(), y)

        metric.accumulate()

        try:
            wandb.log({
                    name + '_accuracy': metric.value,

                    })
            print(name, 'accuracy', metric.value)

        except Exception as e:  # noqa: E722
            print(e)

        return  {
                    'accuracy': metric.value,

                }


def training_process(args, rank, world_size):
    dict_args = vars(args)
    device = int(os.environ['RANK']) % torch.cuda.device_count()


    torch.manual_seed(13)
    torch.cuda.set_device(device)

    views = ['DINOv2', 'CLIP']

    view = views[int(os.environ['RANK']) % len(views)]

    # each view gets its own weights and biases process to monitor resources and performance
    if rank < len(views):
        wandb.init(project=f'MCT e2e with unc {args.dataset}', entity='ai2es',
        name=f"{rank}: {args.train_size}",
        config={'args': dict_args})
                

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

    shape = train[0][0].shape[-1]
    assert train[0][0].shape[-1] == unlbl[0][1].shape[-1] == val[0][0].shape[-1], (view, train[0][0].shape[-1], unlbl[0][0].shape[-1], val[0][0].shape[-1])

    num_classes = 1000

    models = [FCNN(shape, num_classes, [1024, 1024, 1024]) for _ in range(args.members)]

    for i, model in enumerate(models):
        # define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # define learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, min_lr=1e-8)
        # define train loader
        train_loader, _ =  make_loader(train, 1, args.batch_size, shuffle=True)
        # define val loader
        val_loader, _ =  make_loader(val, 1, args.batch_size, shuffle=False)
        # define accuracy logging callbacks
        def train_accuracy_cb(model):
            accuracy_callback(model, train_loader, name=f'{i}_train')
        def val_accuracy_cb(model):
            accuracy_callback(model, val_loader, name=f'{i}_val')


        model = model_run(model, 
              optimizer, 
              scheduler, 
              train_loader, 
              val_loader, 
              criterion=torch.nn.MSELoss(), 
              epochs=args.epochs, 
              patience=args.patience,
              delta=1e-8,
              log_interval=1,
              accumulate=1,
              use_wandb=False,
              checkpoint_path=None,
              metrics=None,
              watch=None,
              watch_mode=min,
              key_map=None,
              callbacks=[train_accuracy_cb, val_accuracy_cb],
              )
        
    # create the ensemble
    ens = Ensemble(models)
    # evaluate the ensemble
    accuracy_callback(ens, val_loader, name='ensemble_val')
    accuracy_callback(ens, train_loader, name='ensemble_train')

    return models


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
    
    parser.add_argument('-m', '--members', type=int, default=12, 
                        help='number of ensembles')
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
    parser.add_argument('--train_size', type=float, default=[0.01],
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
