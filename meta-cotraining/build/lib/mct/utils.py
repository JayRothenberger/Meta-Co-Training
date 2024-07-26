import os
import shutil
import gc

import torch
from torch import nn
from torch.nn import functional as F
import yaml
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import numpy as np
import random
from math import floor, ceil
import pickle
import contextlib
from copy import deepcopy as copy


class LinearProbe(torch.nn.Module):
    def __init__(self, module, size, num_classes, temperature=1.0, mlp=False):
        super().__init__()
        self.m = module

        self.m.train()
        if mlp:
            self.linear = torch.nn.Sequential(
                                                torch.nn.Linear(size, 2048),
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(2048, 1024), 
                                                torch.nn.ReLU(),
                                                torch.nn.Linear(1024, num_classes)
                                                )
        else:
            self.linear = torch.nn.Linear(size, num_classes)

        self.temperature = temperature
    
    def forward(self, x):
        with torch.no_grad():
            x = self.m(x).detach()
        return self.linear(x) * self.temperature


class FPFT(torch.nn.Module):
    def __init__(self, module, temperature=1.0):
        super().__init__()
        self.m = copy(module.m)

        self.m.train()

        self.linear = copy(module.linear)

        self.temperature = temperature
    
    def forward(self, x):
        x = self.m(x)
        return self.linear(x) * self.temperature
    

class FinetunedLinearProbe(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.m = copy(module.m)

        self.m.train()

        self.linear = copy(module.linear)

        self.temperature = module.temperature
    
    def forward(self, x):
        with torch.no_grad():
            x = self.m(x).detach()
        return self.linear(x) * self.temperature


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


class VATLoss(nn.Module):

    def __init__(self, xi=0.1, eps=0.1, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: size of adversarial perturbation 
        :param ip: iterations of projected gradient descent: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():
            pred = F.softmax(model(x), dim=1)

        # prepare random unit tensor
        d = torch.nn.functional.normalize(torch.rand(x.shape)).to(x.device)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = model(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = torch.nn.functional.normalize(d.grad.detach().clone())
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            pred_hat = model(x + r_adv)
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds


def get_embeddings(model, loader, val_loader):
    global_embeds = []
    global_labels = []
    gc.collect()
    with torch.no_grad():
        for i, (images, labels) in enumerate(tqdm(loader)):
            local_embeds  = model(images)
            local_embeds = torch.nan_to_num(local_embeds)
            labels = labels.to(int(os.environ["RANK"]) % torch.cuda.device_count())
            world_embeds = [local_embeds for i in range(int(os.environ["WORLD_SIZE"]))]
            world_labels = [labels for i in range(int(os.environ["WORLD_SIZE"]))]
            dist.all_gather(world_embeds, local_embeds)
            dist.all_gather(world_labels, labels)
            world_embeds = [w.cpu() for w in world_embeds]
            world_labels = [l.cpu() for l in world_labels]
            global_embeds += world_embeds
            global_labels += world_labels

    global_embeds_val = []
    global_labels_val = []
    gc.collect()
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            local_embeds = model(images)
            labels = labels.to(int(os.environ["RANK"]) % torch.cuda.device_count())
            world_embeds = [local_embeds for i in range(int(os.environ["WORLD_SIZE"]))]
            world_labels = [labels for i in range(int(os.environ["WORLD_SIZE"]))]
            dist.all_gather(world_embeds, local_embeds)
            dist.all_gather(world_labels, labels)
            world_embeds = [w.cpu() for w in world_embeds]
            world_labels = [l.cpu() for l in world_labels]
            global_embeds_val += world_embeds
            global_labels_val += world_labels

    return (torch.concat(global_embeds, 0), torch.concat(global_labels, 0), torch.concat(global_embeds_val, 0), torch.concat(global_labels_val, 0))


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class RepeatLoader:
  def __init__(self, loader):
      self.loader = loader
      self.it = iter(loader)
      
  def __iter__(self):
    return self

  def __next__(self):
    try:
        return next(self.it)
    except Exception as e:
        self.it = iter(self.loader)
        return next(self.it)


def create_imagefolder(data, samples, path, transform, new_path=None):
    imgfolder = datasets.ImageFolder(path, transform=transform)
    imgfolder.class_to_idx = data['class_map']
    imgfolder.classes = list(data['class_map'].keys())
    imgfolder.samples = samples

    if new_path is not None:
        imgfolder.root = new_path

    return imgfolder

def create_loader(data, cuda_kwargs, shuffle=False):
    loader_kwargs = {'batch_size': BATCH_SIZE, 'shuffle': shuffle}
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return loader 

def epoch_accuracy(loader_s, loader_t, student, teacher):
    student.eval()
    teacher.eval()
    out_epoch_s = [accuracy(student(L.to(0)), y)[0].cpu().item() for L, y in loader_s]
    out_epoch_t = [accuracy(teacher(L.to(0)), y)[0].cpu().item() for L, y in loader_t]
    student.train()
    teacher.train()
    return sum(out_epoch_s) / len(out_epoch_s), sum(out_epoch_t) / len(out_epoch_t)


def add_to_imagefolder(paths, labels, dataset):
    """
    Adds the paths with the labels to an image classification dataset

    :list paths: a list of absolute image paths to add to the dataset
    :list labels: a list of labels for each path
    :Dataset dataset: the dataset to add the samples to
    """

    new_samples = list(zip(paths, labels))

    dataset.samples += new_samples

    return dataset.samples

# splits the datasets of the two views so that
# the instances inside are still aligned by index
def train_test_split_samples(samples0, samples1, test_size, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    assert test_size > 0 and test_size < 1, \
        'test_size should be a float between (0, 1)'

    assert len(samples0[0]) == len(samples1[0]), \
        f'number of samples in samples0 ({len(samples0)}), samples1 {len(samples1)} are not equal'
    
    idx_samples = list(range(len(samples0[0])))
    idx_test = random.sample(idx_samples, floor(test_size * len(samples0[0])))
    idx_train = list(set(idx_samples) - set(idx_test))

    # convert to np array for convenient array indexing
    #samples0_np = np.stack([np.array(a) for a in samples0])
    #samples1_np = np.stack([np.array(a) for a in samples1])
    
    samples_train0 = samples0[0][idx_train], samples0[1][idx_train]
    samples_test0 = samples0[0][idx_test], samples0[1][idx_test]
    samples_train1 = samples1[0][idx_train], samples1[1][idx_train]
    samples_test1 = samples1[0][idx_test], samples1[1][idx_test]

    assert len(samples_train0[0]) == len(samples_train1[0]), 'sample sizes not equal after split'
    assert len(samples_test0[0]) == len(samples_test1[0]), 'sample sizes not equal after split'

    return samples_train0, samples_test0, samples_train1, samples_test1

class EarlyStopper:
    def __init__(self, stopping_metric, patience):
        self.stopping_metric = stopping_metric
        self.patience = patience
        self.epochs_since_improvement = 0
        self.stop = False 

        self.best_val_loss = float("inf")
        self.best_val_acc = 0

    # TODO perhaps there is a more elegant way to write this
    def is_new_best_metric(self, val_acc, val_loss):
        self.epochs_since_improvement += 1
        if self.stopping_metric == 'loss' and val_loss < self.best_val_loss - 1e-4:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
            return True
        elif self.stopping_metric == 'accuracy' and val_acc > self.best_val_acc + 1e-4:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
            return True
        return False
    
    def early_stop(self):
        if self.epochs_since_improvement > self.patience: 
            return True
        return False


def get_dataset_tensors(from_embed, dataset='IN1K'):
    """
    from_embed - the model from which the embeddings were derived
    dataset - the name of the dataset that was embedded
    """
    with open(f'./{from_embed}_{dataset}_train.ds', 'rb') as fp:
        x_train_embeds, x_train_labels = pickle.load(fp)

    with open(f'./{from_embed}_{dataset}_val.ds', 'rb') as fp:
        x_val_embeds, x_val_labels = pickle.load(fp)

    x_val_embeds = torch.tensor(x_val_embeds)
    x_val_labels = torch.tensor(x_val_labels)

    x_train_embeds = torch.tensor(x_train_embeds)
    x_train_labels = torch.tensor(x_train_labels)

    return (x_train_embeds, x_train_labels, x_val_embeds, x_val_labels), np.unique(x_train_labels).shape[0]


def cascade_round(arr):
    s = 0.0
    arr_cp = np.zeros_like(arr)
    for i, a in enumerate(arr):
        s += a
        if s - (s // 1) > .5:
            arr_cp[i] = ceil(a)
        else:
            arr_cp[i] = floor(a)
    return arr_cp.astype(np.int32)


def subset_npercent(tensors, dataset='IN1K', percent=1, balanced=False):
    percent /= 100.0
    print(percent)
    np.random.seed(13)

    x_train_embeds, x_train_labels, x_val_embeds, x_val_labels = tensors

    if dataset == 'iNat2017':
        unique0, counts0 = np.unique(x_train_labels, return_counts=True)
        keep = unique0[-1010:]
        train_keep = np.hstack([np.where(x_train_labels == l)[0] for l in keep])
        val_keep = np.hstack([np.where(x_val_labels == l)[0] for l in keep])

        x_train_embeds = x_train_embeds[train_keep]
        x_train_labels = x_train_labels[train_keep]

        x_val_embeds = x_val_embeds[val_keep]
        x_val_labels = x_val_labels[val_keep]

    if percent == 0.01 and dataset == 'IN1K':
        with open('./1percent_idx.pkl', 'rb') as fp:
            idx = pickle.load(fp)
    elif percent == 0.1 and dataset == 'IN1K':
        with open('./10percent_idx.pkl', 'rb') as fp:
            idx = pickle.load(fp)
    elif balanced: # create a custom balanced split of percent
        # for count_per_class
        unique, counts = np.unique(x_train_labels, return_counts=True)
        counts_per_class = np.maximum(np.minimum((np.ones_like(counts) * ((x_train_labels.shape[0] / counts.shape[0]) * percent)) - 1.0, counts), 1.0)

        counts_per_class_rounded = np.minimum(cascade_round(counts_per_class), counts)

        while sum(counts_per_class_rounded) < (x_train_labels.shape[0] * percent): # add a single example in the cascade round
            counts_per_class = np.minimum(counts_per_class + (1 / counts.shape[0]), counts)
            counts_per_class_rounded = np.minimum(cascade_round(counts_per_class), counts)

        mask = np.hstack([np.random.choice(np.where(x_train_labels == unique[l])[0], int(counts_per_class_rounded[l]), replace=False)
                            for l in range(unique.shape[0])])

        idx = mask
    
    else: # create a custom stratified split
        unique, counts = np.unique(x_train_labels, return_counts=True)
        count_per_class = percent * counts
        # ok, but this is not exactly n% we will have some rounding to do here
        count_per_class = cascade_round(count_per_class)

        mask = np.hstack([np.random.choice(np.where(x_train_labels == unique[l])[0], count_per_class[l], replace=False)
                            for l in range(unique.shape[0])])

        idx = mask
    

    unlb_idx = list(set(list(range(x_train_labels.shape[0]))) - set(idx))

    unique0, counts0 = np.unique(x_train_labels, return_counts=True)

    x_val_embeds = torch.tensor(x_val_embeds)
    x_val_labels = torch.tensor(x_val_labels)

    x_unlbl_embeds = torch.tensor(x_train_embeds[unlb_idx])
    x_unlbl_labels = torch.tensor(x_train_labels[unlb_idx])

    x_train_embeds = torch.tensor(x_train_embeds[idx])
    x_train_labels = torch.tensor(x_train_labels[idx])

    unique, counts = np.unique(x_train_labels, return_counts=True)

    print(counts)
    print(sum(counts), sum(counts0))
    print(sum(counts) / sum(counts0))

    return x_train_embeds, x_train_labels, x_unlbl_embeds, x_unlbl_labels, x_val_embeds, x_val_labels


def subset_npercent_dataset(ds, percent=1, balanced=True):
    percent /= 100.0
    print(percent)
    np.random.seed(13) 

    x_train_labels = np.array([y for x, y in ds.samples])
    samples = np.array(ds.samples)


    if balanced: # create a custom balanced split of percent
        # for count_per_class
        unique, counts = np.unique(x_train_labels, return_counts=True)
        counts_per_class = np.maximum(np.minimum((np.ones_like(counts) * ((x_train_labels.shape[0] / counts.shape[0]) * percent)) - 1.0, counts), 1.0)

        counts_per_class_rounded = np.minimum(cascade_round(counts_per_class), counts)

        while sum(counts_per_class_rounded) < (x_train_labels.shape[0] * percent): # add a single example in the cascade round
            counts_per_class = np.minimum(counts_per_class + (1 / counts.shape[0]), counts)
            counts_per_class_rounded = np.minimum(cascade_round(counts_per_class), counts)

        mask = np.hstack([np.random.choice(np.where(x_train_labels == unique[l])[0], int(counts_per_class_rounded[l]), replace=False)
                            for l in range(unique.shape[0])])

        idx = mask
    
    else: # create a custom stratified split
        unique, counts = np.unique(x_train_labels, return_counts=True)
        count_per_class = percent * counts
        # ok, but this is not exactly n% we will have some rounding to do here
        count_per_class = cascade_round(count_per_class)

        mask = np.hstack([np.random.choice(np.where(x_train_labels == unique[l])[0], count_per_class[l], replace=False)
                            for l in range(unique.shape[0])])

        idx = mask
    

    unlb_idx = list(set(list(range(x_train_labels.shape[0]))) - set(idx))

    unique0, counts0 = np.unique(x_train_labels, return_counts=True)

    unlbl_samples = samples[unlb_idx]

    train_samples = samples[idx]

    unique, counts = np.unique(x_train_labels, return_counts=True)

    print(len(unlb_idx), len(idx))

    ds, unbl_ds = copy(ds), copy(ds)
    ds.samples = [(x[0], int(x[1])) for x in train_samples]
    unbl_ds.samples = [(x[0], int(x[1])) for x in unlbl_samples]

    return ds, unbl_ds


def make_dataset(from_embed, dataset='IN1K', percent=1, balanced=False):

    tensors, num_classes = get_dataset_tensors(from_embed, dataset=dataset)

    if percent < 100:
        x_train_embeds, x_train_labels, x_unlbl_embeds, x_unlbl_labels, x_val_embeds, x_val_labels = subset_npercent(tensors, dataset=dataset, percent=percent, balanced=balanced)

        train_dataset = torch.utils.data.TensorDataset(x_train_embeds, x_train_labels)
        unlbl_dataset = torch.utils.data.TensorDataset(x_unlbl_embeds, x_unlbl_labels)
        val_dataset = torch.utils.data.TensorDataset(x_val_embeds, x_val_labels)
    else:
        x_train_embeds, x_train_labels, x_val_embeds, x_val_labels = tensors

        train_dataset = torch.utils.data.TensorDataset(x_train_embeds, x_train_labels)
        unlbl_dataset = None
        val_dataset = torch.utils.data.TensorDataset(x_val_embeds, x_val_labels)


    return train_dataset, unlbl_dataset, val_dataset, num_classes


def make_concat_dataset(from_embed, to_embed, dataset='IN1K', percent=1, balanced=False):
    tensors0, num_classes = get_dataset_tensors(from_embed, dataset=dataset)
    tensors1, num_classes = get_dataset_tensors(to_embed, dataset=dataset)
    
    if percent < 100:
        x_train_embeds, x_train_labels, x_unlbl_embeds, x_unlbl_labels, x_val_embeds, x_val_labels = subset_npercent(tensors0, dataset=dataset, percent=percent, balanced=balanced)
        y_train_embeds, y_train_labels, y_unlbl_embeds, y_unlbl_labels, y_val_embeds, y_val_labels = subset_npercent(tensors1, dataset=dataset, percent=percent, balanced=balanced)

        train_dataset = torch.utils.data.TensorDataset(torch.cat((x_train_embeds, y_train_embeds), -1), x_train_labels)
        unlbl_dataset = torch.utils.data.TensorDataset(torch.cat((x_unlbl_embeds, y_unlbl_embeds), -1), x_unlbl_labels)
        val_dataset = torch.utils.data.TensorDataset(torch.cat((x_val_embeds, y_val_embeds), -1), x_val_labels)
    else:
        x_train_embeds, x_train_labels, x_val_embeds, x_val_labels = tensors0
        y_train_embeds, y_train_labels, y_val_embeds, y_val_labels = tensors1

        train_dataset = torch.utils.data.TensorDataset(torch.cat((x_train_embeds, y_train_embeds), -1), x_train_labels)
        unlbl_dataset = None
        val_dataset = torch.utils.data.TensorDataset(torch.cat((x_val_embeds, y_val_embeds), -1), x_val_labels)

    return train_dataset, unlbl_dataset, val_dataset, num_classes


def step_perf(loader_train0, loader_train1, loader_val0, loader_val1, model0, model1, s):
    val_acc0, val_acc1 = epoch_accuracy(loader_val0, loader_val1, model0, model1)
    #val_acc0x, val_acc1x = epoch_accuracy(loader_val1, loader_val0, model0, model1)
    train_acc0, train_acc1 = epoch_accuracy(loader_train0, loader_train1, model0, model1)
    s += 1
    return {'val_acc0': val_acc0,
                    'val_acc1': val_acc1,
                    'train_acc0': train_acc0,
                    'train_acc1': train_acc1}, s

def create_sampler_loader(args, rank, world_size, data, cuda_kwargs={'num_workers': 12, 'pin_memory': True, 'shuffle': False}, shuffle=True):
    
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=shuffle)

    loader_kwargs = {'batch_size': args.batch_size, 'sampler': sampler}
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return sampler, loader   


def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()