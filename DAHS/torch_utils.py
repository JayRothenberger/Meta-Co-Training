import torch
import os
import pickle
import time
import random

def sync_parameters(args, rank, search_space, agent_class):
    # sync parameters between ranks
    path = args.path
    # 1. create the agent
    if rank == 0:
        try:
            agent = agent_class(path, search_space, args)
        except SystemError as e:
            time.sleep(random.randrange(0, 60))
            agent = agent_class(path, search_space, args)
    # 2. generate and broadcast a unique integer - this will specify a path
    # broadcast the integer
    agree = random.randrange(0, 2**32)
    agree = torch.Tensor([agree]).to(rank % torch.cuda.device_count())

    torch.distributed.broadcast(agree, 0)
    torch.distributed.all_reduce(agree)
    agree = int(agree.cpu()[0])

    if rank == 0:
        try:
            os.mkdir(os.path.join(path, f'{agree}'))
        except Exception as e:
            print(e)
        print(path)
        with open(os.path.join(path, f'{agree}/hparams.pkl'), 'wb') as fp:
            pickle.dump(agent, fp)
    else:
        time.sleep(20)

    # load the mutual file
    with open(os.path.join(path, f'{agree}/hparams.pkl'), 'rb') as fp:
        agent = pickle.load(fp)

    return agent