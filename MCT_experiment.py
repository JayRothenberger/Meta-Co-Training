import numpy as np
import torch
import torchvision
import argparse

from torchvision import datasets

import pickle
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from math import floor
import wandb

from models import FCNN
from DAHS.DAHB import DistributedAsynchronousGridSearch
from utils import *
from DAHS.torch_utils import *

from mct import MCT


def c_test(args, rank, world_size, loader0, loader1, model0, model1, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    model0.eval()
    model1.eval()
    with torch.no_grad():
        for batch, ((X0, y0), (X1, y1))  in enumerate(zip(iter(loader0), iter(loader1))):
            X0, y0 = X0.to(device), y0.to(device)
            X1, y1 = X1.to(device), y1.to(device)

            output = torch.nn.Softmax(-1)(model0(X0)) * torch.nn.Softmax(-1)(model1(X1))

            ddp_loss[1] += (output.argmax(1) == y1).type(torch.float).sum().item()
            ddp_loss[2] += len(X0)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*test_acc, test_loss))

    return test_acc

def a_test(args, rank, world_size, loader0, loader1, model0, model1, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    model0.eval()
    model1.eval()
    with torch.no_grad():
        for batch, ((X0, y0), (X1, y1))  in enumerate(zip(iter(loader0), iter(loader1))):
            X0, y0 = X0.to(device), y0.to(device)
            X1, y1 = X1.to(device), y1.to(device)

            output = torch.nn.Softmax(-1)(model0(X0) + model1(X1))

            ddp_loss[1] += (output.argmax(1) == y1).type(torch.float).sum().item()
            ddp_loss[2] += len(X0)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*test_acc, test_loss))

    return test_acc

def b_test(args, rank, world_size, loader0, loader1, model0, model1, device):
    loss_fn = torch.nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    model0.eval()
    model1.eval()
    with torch.no_grad():
        for batch, ((X0, y0), (X1, y1))  in enumerate(zip(iter(loader0), iter(loader1))):
            X0, y0 = X0.to(device), y0.to(device)
            X1, y1 = X1.to(device), y1.to(device)

            output = (torch.nn.Softmax(-1)(model0(X0)) + torch.nn.Softmax(-1)(model1(X1))) / 2.0

            ddp_loss[1] += (output.argmax(1) == y1).type(torch.float).sum().item()
            ddp_loss[2] += len(X0)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*test_acc, test_loss))

    return test_acc


def training_process(args, rank, world_size):
    
    if rank == 0:
        wandb.init(project=f'MCT Bench Grid {args.dataset}', entity='change-this',
        name=f"{rank}: {args.view0} -> {args.view1}",
        config={'args': vars(args)})

    # train0, unlbl0, val0, num_classes = make_concat_dataset(args.view0, 'EsViT', dataset=args.dataset, percent=args.train_size * 100, balanced=args.balanced)
    # train1, unlbl1, val1, num_classes = make_concat_dataset(args.view1, 'MAE', dataset=args.dataset, percent=args.train_size * 100, balanced=args.balanced)

    train0, unlbl0, val0, num_classes = make_dataset(args.view0, dataset=args.dataset, percent=args.train_size * 100)
    train1, unlbl1, val1, num_classes = make_dataset(args.view1, dataset=args.dataset, percent=args.train_size * 100)
    
    model0 = FCNN(train0[0][0].shape[-1], num_classes, [1024, 1024, 1024], nn.BatchNorm1d, nn.ReLU, dropout=0.2).to(0)
    model1 = FCNN(train1[0][0].shape[-1], num_classes, [1024, 1024, 1024], nn.BatchNorm1d, nn.ReLU, dropout=0.2).to(0)
    print(train0[0][0].shape[-1], train1[0][0].shape[-1])

    optimizer0 = optim.Adam(model0.parameters(), lr=args.learning_rate)
    optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)

    scheduler0 = ReduceLROnPlateau(optimizer0, 'max')
    scheduler1 = ReduceLROnPlateau(optimizer1, 'max')

    loss_fn = nn.CrossEntropyLoss()
    vat_loss = VATLoss()

    # Instantiate samplers and get DataLoader objects
    sampler_train0, loader_train0 = create_sampler_loader(args, rank, world_size, train0)
    sampler_unlbl0, loader_unlbl0 = create_sampler_loader(args, rank, world_size, unlbl0)
    sampler_val0, loader_val0 = create_sampler_loader(args, rank, world_size, val0, shuffle=False)

    sampler_train1, loader_train1 = create_sampler_loader(args, rank, world_size, train1)
    sampler_unlbl1, loader_unlbl1 = create_sampler_loader(args, rank, world_size, unlbl1)
    sampler_val1, loader_val1 = create_sampler_loader(args, rank, world_size, val1, shuffle=False)

    stopper0 = EarlyStopper(stopping_metric='accuracy', patience=args.patience)
    stopper1 = EarlyStopper(stopping_metric='accuracy', patience=args.patience)

    states = {
        'model0_state': model0.state_dict(), 
        'optimizer0_state': optimizer0.state_dict(),
        'model1_state': model1.state_dict(),
        'optimizer1_state': optimizer1.state_dict()}

    s = 0
    for epoch in range(args.epochs):

        sampler_train1.set_epoch(epoch)
        sampler_train0.set_epoch(epoch)
        sampler_unlbl1.set_epoch(epoch)
        sampler_unlbl0.set_epoch(epoch)
        
        loss = torch.nn.CrossEntropyLoss()
        if epoch < 60:
            for e, (L_t, Y_t), (L_s, Y_s) in tqdm(zip(range(len(loader_unlbl0)), iter(RepeatLoader(loader_train0)), iter(RepeatLoader(loader_train1)))):
                teacher_out = model0(L_t.to(0))
                teacher_loss_sup = loss(teacher_out, Y_t.to(0))
                teacher_loss_sup.backward()
                # torch.nn.utils.clip_grad_norm_(model0.parameters(), 1.0)
                optimizer0.step()
                optimizer0.zero_grad()

                student_out = model1(L_s.to(0))
                student_loss_sup = loss(student_out, Y_s.to(0)) 
                student_loss_sup.backward()
                # torch.nn.utils.clip_grad_norm_(model1.parameters(), 1.0)
                optimizer1.step()
                optimizer1.zero_grad()
                if e % 10 == 0:
                    if rank == 0:
                        model0.eval()
                        model1.eval()
                        d, s = step_perf(loader_train0, loader_train1, loader_val0, loader_val1, model0, model1, s)
                        d['a_acc'] = a_test(args, 0, 1, loader_val0, loader_val1, model0, model1, 0)
                        d['b_acc'] = b_test(args, 0, 1, loader_val0, loader_val1, model0, model1, 0)
                        d['c_acc'] = c_test(args, 0, 1, loader_val0, loader_val1, model0, model1, 0)

                        scheduler0.step(d['val_acc0'])
                        scheduler1.step(d['val_acc1'])

                        wandb.log(d, step=s)
                        model0.train()
                        model1.train()

                        if stopper0.is_new_best_metric(d['val_acc0'], float('inf')):
                            states['model0_state'] = model0.state_dict()
                            states['optimizer0_state'] = optimizer0.state_dict()

                        if stopper1.is_new_best_metric(d['val_acc1'], float('inf')): 
                            states['model1_state'] = model1.state_dict()
                            states['optimizer1_state'] = optimizer0.state_dict()

        else:
            for e, (U_t, _), (U_s, _), (L_t, Y_t), (L_s, Y_s) in tqdm(zip(range(len(loader_unlbl0)), iter(loader_unlbl0), iter(loader_unlbl1), iter(RepeatLoader(loader_train0)), iter(RepeatLoader(loader_train1)))):

                gc.collect()
                MCT(U_t.to(0), U_s.to(0), L_t.to(0), L_s.to(0), Y_t.to(0), Y_s.to(0), model0, model1, optimizer0, optimizer1, loss=torch.nn.CrossEntropyLoss(), supervised=True, approx=False, previous_params=True)
                # U_t, U_s, L_t, L_s, Y_t, Y_s, student, teacher, student_optimizer, teacher_optimizer, loss=torch.nn.CrossEntropyLoss(), supervised=False, approx=False, previous_params=True
                if e % 2 == 0:
                    if rank == 0:
                        model0.eval()
                        model1.eval()
                        d, s = step_perf(loader_train0, loader_train1, loader_val0, loader_val1, model0, model1, s)
                        d['a_acc'] = a_test(args, 0, 1, loader_val0, loader_val1, model0, model1, 0)
                        d['b_acc'] = b_test(args, 0, 1, loader_val0, loader_val1, model0, model1, 0)
                        d['c_acc'] = c_test(args, 0, 1, loader_val0, loader_val1, model0, model1, 0)

                        scheduler0.step(d['val_acc0'])
                        scheduler1.step(d['val_acc1'])

                        wandb.log(d, step=s)
                        model0.train()
                        model1.train()
                        
                        if stopper0.is_new_best_metric(d['val_acc0'], float('inf')):
                            states['model0_state'] = model0.state_dict()
                            states['optimizer0_state'] = optimizer0.state_dict()
                        else:
                            model0 = FCNN(train0[0][0].shape[-1], num_classes, [1024, 1024, 1024], nn.BatchNorm1d, nn.ReLU, dropout=0.2).to(0)
                            model0.load_state_dict(states['model0_state'])
                            optimizer0 = optim.Adam(model0.parameters(), lr=args.learning_rate)

                        if stopper1.is_new_best_metric(d['val_acc1'], float('inf')): 
                            states['model1_state'] = model1.state_dict()
                            states['optimizer1_state'] = optimizer0.state_dict()
                        else:
                            model1 = FCNN(train1[0][0].shape[-1], num_classes, [1024, 1024, 1024], nn.BatchNorm1d, nn.ReLU, dropout=0.2).to(0)
                            model1.load_state_dict(states['model1_state'])
                            optimizer1 = optim.Adam(model1.parameters(), lr=args.learning_rate)

                        if stopper0.early_stop() and stopper1.early_stop():
                            break

    return states, stopper0.best_val_acc


def main(args, rank, world_size):
    setup(rank, world_size)

    search_space = ['view0', 'view1', 'train_size', 'balanced']

    agent = sync_parameters(args, rank, search_space, DistributedAsynchronousGridSearch)

    args = agent.to_namespace(agent.combination)

    if (args.view0, args.view1) not in [('DINOv2', 'SwAV'), ('DINOv2', 'MAE'), ('DINOv2', 'EsViT'), ('DINOv2', 'CLIP'),
                                         ('CLIP', 'SwAV'), ('CLIP', 'EsViT'), ('CLIP', 'MAE'),
                                         ('EsViT', 'SwAV'), ('EsViT', 'MAE'),
                                         ('SwAV', 'MAE'),]:
        agent.save_checkpoint(states)
        agent.finish_combination(-1)
        exit()

    states, metric = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)

    print('cleanup')
    cleanup()


        
def create_parser():
    parser = argparse.ArgumentParser(description='MCT benchmark')
    
    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=4096, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-p', '--patience', type=int, default=50, 
                        help='patience for training')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=4096, 
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('--view0', type=str, default=['DINOv2', 'CLIP', 'SwAV', 'EsViT', 'MAE'], metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--view1', type=str, default=['DINOv2', 'CLIP', 'SwAV', 'EsViT', 'MAE'], metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--dataset', type=str, default='IN1K', metavar='e',
                        help='embeddings over which to compute the distances')
    parser.add_argument('--path', type=str, default='./Grid_MCT_IN1K',
                        help='path for hparam search directory')
    parser.add_argument('--train_size', type=float, default=[0.1, 0.01],
                        help='size of the training set (%)')
    parser.add_argument('--balanced', type=bool, default=[False, True], 
                        help='Balanced dataset subsetting if true, else stratified sampling')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    main(args, rank, world_size)
