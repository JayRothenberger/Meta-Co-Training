import torch
from tqdm import tqdm
import gc
import os
from copy import deepcopy as copy
import wandb
import warnings


def train(model, 
          criterion, 
          optimizer, 
          loader, 
          dtype=torch.float32, 
          accumulate=1, 
          silent=False):
    
    model.train()
    model = model.type(dtype)
    loss = criterion
    device = torch.cuda.current_device()

    ddp_loss = torch.zeros(2).to(device)

    for i, (x, y) in tqdm(enumerate(loader)):

        with torch.autocast(device_type="cuda", dtype=dtype):
            out = model(x.type(dtype).to(device))
            l = loss(out, y.to(device)) / accumulate
            l.backward()

        ddp_loss[0] += l
        ddp_loss[1] += x.shape[0]

        if i % accumulate == accumulate - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    torch.distributed.all_reduce(ddp_loss, op=torch.distributed.ReduceOp.AVG)

    if not silent:
        print('\nTrain set: Avg. loss: {:.4f}\n'.format(ddp_loss[0] / ddp_loss[1]))

    return ddp_loss[0] / ddp_loss[1]

@torch.no_grad()
def test(network, 
         criterion, 
         test_loader, 
         silent=False,
         metrics=None,
         dtype=torch.float32):
    
    network.eval()
    network = network.type(dtype)
    device = torch.cuda.current_device()

    test_metrics = dict()
    test_metrics['loss'] = 0

    ddp_loss = torch.zeros(2).to(device)

    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            
            ddp_loss[1] += x.shape[0]

            with torch.autocast(device_type="cuda", dtype=dtype):
                output = network(x.type(dtype).to(device))

            for metric in metrics:
                metrics[metric].step(output, y.to(device))

            ddp_loss[0] += criterion(output, y.to(device)).item()

        torch.distributed.all_reduce(ddp_loss, op=torch.distributed.ReduceOp.AVG)

        test_metrics['loss'] = ddp_loss[0] / ddp_loss[1]

        if not silent:
            print('\nTest set: Avg. loss: {:.4f}\n'.format(test_metrics['loss']))
            
    return test_metrics

def model_run(model, 
              optimizer, 
              scheduler, 
              train_loader, 
              val_loader, 
              criterion=torch.nn.MSELoss(), 
              epochs=75, 
              patience=10,
              delta=1e-8,
              log_interval=1,
              accumulate=1,
              use_wandb=False,
              checkpoint_path=None,
              metrics=None,
              watch=None,
              watch_mode=min,
              dtype=torch.float32,
              callbacks=[],
              ):
    """
    train a model for a number of epochs and return the version with the best validation accuracy


    Parameters
    ----------
    mae_model : torch.nn.Module
        the model to train
    optimizer : torch.optim.Optimizer
        An optimizer for the parameters in the mae_model
    scheduler : torch.optim.LearningRateScheduler
        A learning rate scheduler for the optimizer
    train_loader: torch.utils.data.DataLoader
        A dataloader object which generates batches used to train the mae_model
    val_loader: torch.utils.data.DataLoader
        A dataloader object which generates batches uses to validate the performance of the mae_model
    epochs: int
        Maximum number of training epochs
    patience: int
        Number of consecutive epochs without improvement to continue training before early stopping
    delta: float
        Tolerance threshold for improvement.  Changes in the watch metric of magnitude less than this value 
        are not considered improvements.
    log_interval: int
        Number of epochs after which to compute validation metrics
    accumulate: int
        Number of batches to accumulate gradients over during gradient accumulation
    use_wandb: bool
        Whether or not to log metrics to the current weights and biases process
    checkpoint_path: PathLike or str
        Path to save the checkpoints to each epoch
    metrics: dict[str, Metric]
        A mapping from names of metrics to metric objects that will be evaluated during validation and recorded
        during training.
    watch: Union[None|str]
        None or name of the metric to use for early stopping and learning rate scheduling.  If None, loss is used.
    watch_mode: Union[min|max]
        Whether to minimize or maximize the watch metric, default is min.
    key_map: Callable
        A function that maps the input to a dictionary with keys 'instance' and 'target'
    callbacks: list[Callable]
        A list of functions to call at the end of each epoch whose only required argument is the model


    Returns
    -------
    torch.nn.Module
        The trained model with the best validation performance across all epochs
    """
    loss = watch_mode([float('inf'), float('-inf')])
    loss = float('inf') if float('inf') != loss else float('-inf')

    watch = watch if watch is not None else 'loss'
    metrics = metrics if metrics is not None else dict()

    if 'loss' in metrics:
        warnings.warn('loss key will be overwritten to use loss function')

    if checkpoint_path is None:
        warnings.warn('checkpoint path is None, so checkpoint will not be saved')

    state = None

    epochs_since_improvement = 0

    for epoch in range(epochs):
        print(f'{epoch + 1}/{epochs}')
        train(model, criterion, optimizer, train_loader, accumulate=accumulate, dtype=dtype)
        gc.collect()

        if epoch % log_interval == log_interval - 1:
            epoch_metrics = test(model, criterion, val_loader, metrics=metrics, dtype=dtype)
            
            for fn in callbacks:
                fn(model)

            scheduler.step(epoch_metrics[watch])

            if watch_mode(epoch_metrics[watch], loss) != loss:
                torch.distributed.barrier()
                state = model.state_dict()

                if not (-delta < (loss - epoch_metrics[watch]) < delta):
                    epochs_since_improvement = 0

                loss = epoch_metrics[watch]

                if os.environ['RANK'] == 0:
                    if checkpoint_path is not None:
                        with open(checkpoint_path, 'wb') as fp:
                            torch.save(state, fp)

                if use_wandb:
                    wandb.log(epoch_metrics)
            
            if epochs_since_improvement > patience:
                break
        
        epochs_since_improvement += 1

    model.load_state_dict(state)
    return model