import numpy as np
import torch
import torchvision
import os
import time
from mct.utils import *
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler
import gc
import wandb
import pickle
import time
import warnings

def epoch_model_accuracy(loader, model):
    model.eval()

    out_epoch = [accuracy(model(L.to(int(os.environ['RANK']) % torch.cuda.device_count())), y)[0].item() for L, y in loader]
    model.train()

    return torch.tensor(sum(out_epoch) / len(out_epoch)).to(int(os.environ['RANK']) % torch.cuda.device_count())


class DiffAllGather(torch.autograd.Function):
    """
    The normal all-gather does not support autograd.  This one does.
    """
    @staticmethod
    def forward(ctx, tensor):
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        torch.distributed.barrier()
        
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grad_outs):
        print(os.environ["RANK"], "grads out length:", len(grad_outs))
        grad_outs = torch.stack(grad_outs)
        print(os.environ["RANK"], "grads out:", grad_outs)
        dist.all_reduce(grad_outs, op=torch.distributed.ReduceOp.SUM)
        print(os.environ["RANK"], "grads out reduced:", grad_outs)
        torch.distributed.barrier()
        return grad_outs[dist.get_rank()]


def make_loader(view, num_models, batch_size):
    kwargs={'num_workers': 12, 'pin_memory': True, 'shuffle': False}

    sampler = DistributedSampler(view, rank=int(os.environ['RANK']) // num_models, num_replicas=int(os.environ['WORLD_SIZE']) // num_models, shuffle=True)

    loader_kwargs = {'batch_size': batch_size, 'sampler': sampler}
    loader_kwargs.update(kwargs)

    loader = DataLoader(view, **loader_kwargs)

    return loader, sampler


class MetaCoTrainingModel(torch.nn.Module):
    """
    wrapper class for meta co-training models

    TODO: checkpointing (maybe)
    """

    def __init__(self, models, accum_steps=1):
        super().__init__()
        self.models = models
        self.groups = []
        self.cpu_grads = []

        self.loss_initial = 0.0
        self.loss_final = 0.0
        self.self_loss_grads = []
        self.loss_initial_grads = []
        self.grads1 = []
        self.loss_final_grads = []
        self.supervised_grads = []
        self.autocast_type = torch.bfloat16

        self.best_val_accs = []

        self.accum_steps = accum_steps

        # if I could use process groups, this is how I would construct them
        #if int(os.environ['WORLD_SIZE']) % len(self.models) == 0:
        #    for j in range(len(self.models)):
        #        print(j, list(range(int(os.environ['WORLD_SIZE'])))[j::len(self.models)])
        #        self.groups.append(torch.distributed.new_group(list(range(int(os.environ['WORLD_SIZE'])))[j::len(self.models)]))
        #else:
        #    self.groups.append(list(range(int(os.environ['WORLD_SIZE']))))


    def all_gather(self, tensor):
        return DiffAllGather.apply(tensor)


    def rank_pool_reduce(self, tensor, op=torch.distributed.ReduceOp.AVG):
        """
        unfortunately we cannot use nccl to perform reductions across multiple process groups 
        https://discuss.pytorch.org/t/nccl-backend-not-safe-for-processgroups/182941
        it does cause deadlocks the way I intended to use it here, but perhaps if explicit attention was paid
        to synchronization then there would be some improvement in speed, and definitely in network usage here.

        in this case all tensors are gathered and the ones we do not need are discarded.
        """
        tensors = self.all_gather(tensor)

        tensors = [tensor for i, tensor in enumerate(tensors) if i % len(self.models) == int(os.environ['RANK']) % len(self.models)]
        return sum(tensors) / len(tensors)

    def other_pool_reduce(self, tensor):
        torch.distributed.barrier()
        tensors = [torch.zeros_like(tensor) for i in range(int(os.environ['WORLD_SIZE']))]
        # blocking operation that gathers the tensors from tensor across the ranks coresp to the models assigned to other ranks
        if int(os.environ['WORLD_SIZE']) % len(self.models) == 0:
            # in this case each tensor in the list is from a rank pool, and we need only discard the local copy
            torch.distributed.all_gather(tensors, tensor, async_op=False)
            for i in range(len(tensors))[::-1]:
                if i % len(self.models) == int(os.environ['RANK']) % len(self.models):
                    tensors.pop(i)
            tensors = sum(tensors) / len(tensors)
        else:
            # in this case the result is poorly defined and should probably be avoided
            torch.distributed.all_gather(tensors, tensor, async_op=False)
            tensors.pop(int(os.environ['RANK']))
            tensors = sum(tensors) / len(tensors)
        torch.distributed.barrier()
        return tensors

    def reduce_weights(self):
        """
        gathers all of the weights and puts them on the zero rank device in the zero rank process.
        """
        # agree on a random number so multiple simultaneous runs can occur on the same filesystem
        # very unlikely to encounter a collision, but could replace with some other unique value
        # slurm job ID is a good unique value
        agree = random.randrange(0, 2**32)
        agree = torch.Tensor([agree]).to(int(os.environ["RANK"]) % torch.cuda.device_count())

        torch.distributed.broadcast(agree, 0)
        torch.distributed.all_reduce(agree)
        agree = int(agree.cpu()[0])

        rankpath = f'./MCT_weights/{agree}_{os.environ["RANK"]}.pt'

        try:
            os.mkdir('./MCT_weights')
        except:
            pass

        start = time.time()
        time.sleep(1)
        if int(os.environ['RANK']) < len(self.models):
            with open(rankpath, 'wb') as fp:
                pickle.dump(self.models[int(os.environ['RANK'])].state_dict(), fp)

        # if int(os.environ['RANK']) == 0:
        for i in range(len(self.models)):
            rankpath = f'./MCT_weights/{agree}_{i}.pt'

            # while the file has not been written
            while not os.path.isfile(rankpath):
                time.sleep(0.01)
                start = time.time()
            # we just wrote the file, so ctime needs to change
                
            # while it has not finished writing
            while os.path.getctime(rankpath) < start:
                time.sleep(1)
            
            while True:
                try:
                    with open(rankpath, 'rb') as fp:
                        self.models[i].load_state_dict(pickle.load(fp))
                        self.models[i].to(0)
                        break
                except Exception as e:
                    print(f'{os.environ["RANK"]} error reducing weights: {e}')
                    continue

    def co_accuracy(self, loaders):
        for model in self.models:
            model.eval().to(int(os.environ['RANK']) % torch.cuda.device_count())

        loader = zip(*[iter(loader) for loader in loaders])

        out_epoch = [accuracy(self([X for X, y in L]), [y for X, y in L][0])[0].item() for L in loader]
        
        for model in self.models:
            model.train()

        return torch.tensor(sum(out_epoch) / len(out_epoch)).to(int(os.environ['RANK']) % torch.cuda.device_count())

    def forward(self, x):
        assert len(x) == len(self.models)
        return sum([model(l.to(int(os.environ['RANK']) % torch.cuda.device_count())) for model, l in zip(self.models, x)])

    def train(self, epochs, warmup, train_views, unlbl_views, val_views, test_views, checkpoint_path, optimizer=Adam, batch_size=1024, lr=1e-4, lr_scheduler=ReduceLROnPlateau, patience=1, amp=True, use_wandb=False, log_interval=1, approx=True, supervised=True):
        """
        epochs: number of epochs to train after warmup
        warmup: warmup epochs to train without MCT loss
        train_views: datasets of training views
        unlbl_views: datasets of unlabeled views
        val_views: datasets of val views
        optimizer: optimizer class
        batch_size: training batch size
        lr: learning rate
        lr_scheduler: learning rate scheduler
        patience: patience for early stopping
        amp: whether or not to use automatic mixed precision
        wandb: whether or not to log to wandb

        if the number of gpus is not divisible by the number of models then we are going to do normal DDP on each model
        (this should raise a warning TODO, requires a distributed sampler)

        otherwise we are putting each of them on a different GPU and computing their batches simultaneously 
        """
        # TODO: add early stopping
        # only log to wandb if the option is set
        print('training')

        optimizers = []
        mct_optimizers = []
        schedulers = []
        stoppers = []
        samplers = []
        samplers_unlbl = []
        scalers = [GradScaler() for model in self.models]

        mct_scaler = torch.cuda.amp.GradScaler()

        assert len(train_views) == len(val_views) == len(unlbl_views), f"number of views must be the same for train, unlabeled, val but got: {len(train_views)}, {len(unlbl_views)}, {len(val_views)}"
        assert int(os.environ['WORLD_SIZE']) % len(self.models) == 0, f"number of models must be divisible by number of gpus"


        loss = nn.CrossEntropyLoss()

        for i in range(len(self.models)):
            stoppers.append(EarlyStopper(stopping_metric='accuracy', patience=patience))

        # then we need to send each model to their own ranks, so we send only the model for our rank.
        self.models[int(os.environ['RANK']) % len(self.models)].to(int(os.environ['RANK']) % torch.cuda.device_count())

        for model in self.models:
            model.train()
            optimizers.append(optimizer(model.parameters(), lr=lr))
            mct_optimizers.append(optimizer(model.parameters(), lr=lr / 8.0))
        
        if lr_scheduler is not None:
            for optimizer in optimizers:
                schedulers.append(lr_scheduler(optimizer, 'max', factor=0.5, patience=16))
        
        for i in range(len(train_views)):

            train_views[i], sampler = make_loader(train_views[i], len(self.models), batch_size)

            unlbl_views[i], sampler_unlbl = make_loader(unlbl_views[i], len(self.models), batch_size)

            val_views[i], _ = make_loader(val_views[i], len(self.models), batch_size)

            test_views[i], _ = make_loader(test_views[i], len(self.models), batch_size)
            
            samplers.append(sampler)
            samplers_unlbl.append(sampler_unlbl)
            
        self.val_views = val_views
        self.test_views = test_views

        if int(os.environ['WORLD_SIZE']) % len(self.models) == 0:
            # at this point we can filter out those loaders we do not care about
            # those are the loaders that are not \equiv int(os.environ['RANK']) (mod len(self.models))
            # do not be confused.  This leaves only one view.

            train_views = [train_views[int(os.environ['RANK']) % len(train_views)]]
            unlbl_views = [unlbl_views[int(os.environ['RANK']) % len(unlbl_views)]]
            val_views = [val_views[int(os.environ['RANK']) % len(val_views)]]
            test_views = [test_views[int(os.environ['RANK']) % len(val_views)]]


        states = {f'model{i}_state': model.state_dict() for i, model in enumerate(self.models)}
        states.update({f'optimizer{i}_state': optimizer.state_dict() for i, model in enumerate(optimizers)})

        self.cpu_grads = [None for param in self.models[int(os.environ['RANK']) % len(self.models)].parameters()]

        self.s = 0
        for epoch in range(epochs):
            print(epochs, epoch)
            for sampler in samplers:
                sampler.set_epoch(epoch)
                sampler_unlbl.set_epoch(epoch)
            d = dict()

            scaler = torch.cuda.amp.GradScaler()

            assert int(os.environ['WORLD_SIZE']) % len(self.models) == 0, 'number of GPUs must be divisible my number of models'

            if epoch < warmup:
                for e, L in tqdm(zip(*[range(len(unlbl_views[0])), zip(*[iter(v) for v in train_views])])):
                    gc.collect()
                    self.s += 1
                    i = int(os.environ['RANK']) % len(self.models)
                    model = self.models[i].to(int(os.environ['RANK']) % torch.cuda.device_count())
                    stopper = stoppers[i]
                    optimizer = optimizers[i]
                    scheduler = schedulers[i]
                    X, y = L[0]
                    with torch.autocast(device_type="cuda", dtype=self.autocast_type):
                        out = model(X.to(int(os.environ['RANK']) % torch.cuda.device_count()))
                        loss_sup = loss(out, y.to(int(os.environ['RANK']) % torch.cuda.device_count())) / self.accum_steps

                    scaler.scale(loss_sup).backward()
                    
                    if self.s % self.accum_steps == self.accum_steps - 1:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        scaler.step(optimizer)
                        scaler.update()

                        optimizer.zero_grad()
                        

                    if self.s % log_interval == 0:
                        
                        model.eval()

                        d[f'val_acc{i}'] = self.rank_pool_reduce(epoch_model_accuracy(val_views[0], model))
                        d[f'test_acc{i}'] = self.rank_pool_reduce(epoch_model_accuracy(test_views[0], model))
                        torch.distributed.barrier()

                        self.reduce_weights()
                        if int(os.environ["RANK"]) == 0:
                            for j, m in enumerate(self.models):
                                with open(f'{checkpoint_path}_{j}', 'wb') as fp:
                                    pickle.dump(self.models[j], fp)

                        scheduler.step(d[f'val_acc{i}'])
                        if int(os.environ['RANK']) < len(self.models):
                            wandb.log(d, step=self.s)

                            if stopper.is_new_best_metric(d[f'val_acc{i}'], float('inf')):
                                states[f'model{i}_state'] = model.state_dict()

            else:
                # load the best performing model in warmup
                if epoch == warmup:
                    i = int(os.environ['RANK']) % len(self.models)
                    model = self.models[i]
                    model.load_state_dict(states[f'model{i}_state'])

                for e, (U, X) in tqdm(zip(*[range(len(unlbl_views[0])), zip(zip(*[iter(RepeatLoader(v)) for v in unlbl_views]), zip(*[iter(RepeatLoader(v)) for v in train_views]))])):
                    self.s += 1
                    gc.collect()
                    i = int(os.environ['RANK']) % len(self.models)
                    model = self.models[i]
                    stopper = stoppers[i]
                    optimizer = mct_optimizers[i]
                    scheduler = schedulers[i]
                    scaler = scalers[i]

                    U, _ = U[0]
                    X, y = X[0]

                    self.mct(U.to(int(os.environ['RANK']) % torch.cuda.device_count()), X.to(int(os.environ['RANK']) % torch.cuda.device_count()), y.to(int(os.environ['RANK']) % torch.cuda.device_count()),
                                model, optimizer, mct_scaler, loss=loss, supervised=supervised, approx=approx)

                    if self.s % log_interval == 0:
                        model.eval()
                        d[f'val_acc{i}'] = self.rank_pool_reduce(epoch_model_accuracy(val_views[0], model))
                        d[f'test_acc{i}'] = self.rank_pool_reduce(epoch_model_accuracy(test_views[0], model))
                        torch.distributed.barrier()
                        self.reduce_weights()

                        if int(os.environ['RANK']) == 0:
                            for j, m in enumerate(self.models):
                                with open(f'{checkpoint_path}_{j}', 'wb') as fp:
                                    pickle.dump(self.models[j], fp)

                        torch.distributed.barrier()
                        d[f'c_acc'] = self.co_accuracy(self.val_views)
                        d[f'c_acc_test'] = self.co_accuracy(self.test_views)

                        scheduler.step(d[f'val_acc{i}'])
                        if int(os.environ['RANK']) < len(self.models):
                            wandb.log(d, step=self.s)

                            if stopper.is_new_best_metric(d[f'val_acc{i}'], float('inf')):
                                states[f'model{i}_state'] = model.state_dict()
                            elif stopper.epochs_since_improvement > 5:
                                model.load_state_dict(states[f'model{i}_state'])
            

        self.best_val_accs = [stopper.best_val_acc for stopper in stoppers]

        return states

    def accumulate_gradient(self, model, into):
        # put the gradient in the list
        if into:
            for param, grads in zip(model.parameters(), into):
                if param.grad is not None:
                    grads += param.grad.detach().clone()
                    param.grad = None
        else:
            for param in model.parameters():
                if param.grad is not None:
                    into.append(param.grad.detach().clone())
                else:
                    into.append(None)

    def store_gradient(self, model, into, scale_factor=1):
        # put the gradient in the model
        if into:
            for i, (param, grads) in enumerate(zip(model.parameters(), into)):
                if param.grad is not None:
                    param.grad = grads / scale_factor
                    into[i] = None
            
            into.clear()
        

    def mct(self, U, X, y, model, optimizer, scaler, loss=torch.nn.CrossEntropyLoss(), supervised=True, approx=False):
        accum_steps = self.accum_steps
        dist.broadcast(U, 0)

        #if not torch.equal(tensors[0], tensors[1]):
        #    print('all gather did not yeild identical tensors, badness: ', (tensors[0] - tensors[1]).mean())
        #    return
        
        mct_length = (2 * accum_steps) + 1

        if self.s % mct_length < accum_steps:
            # persistent variables for this function to accumulate gradients:
            if self.s % mct_length == 0:
                # these are reset on the 0 % (2 * accum_steps)
                self.loss_initial = 0.0
                self.loss_final = 0.0
                self.self_loss_grads = []
                self.loss_initial_grads = []
                self.grads1 = []
                self.loss_final_grads = []
                self.supervised_grads = []

            device = int(os.environ["RANK"]) % torch.cuda.device_count()

            self.store_gradient(model, self.self_loss_grads)

            with torch.autocast(device_type="cuda", dtype=self.autocast_type):

                SPL_o = model(U)

                # gather the PLs from the other ranks
                SPL = self.other_pool_reduce(SPL_o).detach()
                assert not torch.equal(SPL_o, torch.zeros_like(SPL_o))

                PL = torch.tensor([
                    np.random.choice(np.arange(SPL.shape[-1]),
                                                 None, 
                                                 False, 
                                                 torch.nn.Softmax(-1)(
                                                            torch.clamp(
                                                            torch.nan_to_num(
                                                                        SPL.type(torch.float32)
                                                                    ), 
                                                                    -2**14, 2**14
                                                                    )
                                                                ).detach().cpu().numpy()[xi]
                                    ) 
                                for xi in range(SPL.shape[0])
                                ]).to(device) # sample from the SPL distribution (taking the argmax may be more stable if calibration is poor)

                self_loss = loss(SPL_o, PL)
                

            scaler.scale(self_loss).backward()

            torch.distributed.barrier()
            # accumulate the self loss grads over the number of steps
            self.accumulate_gradient(model, self.self_loss_grads)


            with torch.autocast(device_type="cuda", dtype=self.autocast_type):
                if approx:
                    initial_output = model(X)
                    self.loss_initial += self.other_pool_reduce(loss(initial_output, y).detach().clone())

            optimizer.zero_grad()
            gc.collect()

            self.store_gradient(model, self.loss_initial_grads)

            with torch.autocast(device_type="cuda", dtype=self.autocast_type):
                initial_output_u = model(U)

                loss_initial_pl = loss(initial_output_u, PL)
            
            optimizer.zero_grad()
            scaler.scale(loss_initial_pl).backward()
            # loss_initial_pl.backward()

            if self.s % mct_length == (accum_steps - 1):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not approx:
                    self.accumulate_gradient(copy(model), self.grads1)

                # update the student parameters based on pseudo labels from teacher
                print('student step')
                scaler.step(optimizer)
                # optimizer.step()
                scaler.update()
                optimizer.zero_grad()

        elif self.s % mct_length < (2 * accum_steps):

            with torch.autocast(device_type="cuda", dtype=self.autocast_type):
                if approx:
                    with torch.no_grad():
                        final_output = model(X)
                        loss_final = loss(final_output.detach(), y)
                else:
                    self.store_gradient(model, self.loss_final_grads)
                    final_output = model(X)
                    loss_final = loss(final_output, y)

            if approx:
                scaler.scale(loss_final)
                self.loss_final += self.other_pool_reduce(loss_final.detach().clone())
            else:
                scaler.scale(loss_final).backward()

                self.loss_final += self.other_pool_reduce(loss_final.detach().clone())

                if self.s % mct_length == (2 * accum_steps) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.accumulate_gradient(model, self.loss_final_grads)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=self.autocast_type):
                if supervised:  # optionally compute the supervised loss
                    self.store_gradient(model, self.supervised_grads)
                    out = model(X)
                    loss_sup = loss(out, y)
                    # TODO: this should be reduced among the members of the process group
                    scaler.scale(loss_sup).backward()
                    self.accumulate_gradient(model, self.supervised_grads)
            
            optimizer.zero_grad()

        else:
            if (not self.self_loss_grads) or (self.loss_final == 0):
                warnings.warn('attempting MCT step without self_loss_grads or zero loss_final, it looks like s was not set correctly...')
                return

            if approx:
                # https://github.com/google-research/google-research/issues/536
                # h is approximable by: student_loss_final - loss(student_initial(L), Y) where student_initial is before the gradient update for U'
                h = (self.loss_initial - self.loss_final) / accum_steps
            else:
                # for correctness, I include instead the theoretically correct computation of h
                h = sum([(self.other_pool_reduce((param.grad.detach() / accum_steps) * (grads / accum_steps))).sum() for param, grads in zip(self.loss_final_grads,  self.grads1) if param is not None and grads is not None])
            

            self.store_gradient(model, self.supervised_grads, scale_factor=accum_steps)
            
            # compute the MPL update based on the original parameters
            for param, grad in zip(model.parameters(), self.self_loss_grads):
                if param.grad is not None and grad is not None:
                    param.grad += h * grad
                elif grad is not None:
                    param.grad = h * grad
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
