"""
logistics/dahps.py by Jay Rothenberger (jay.c.rothenberger@gmail.com)

Distributed Asynchronous Hyper Parameter Search is a style of hyperparameter optimization that
maintains a persistent state, in this case in a external database, which is used to coordinate
model runs across multiple different machines and devices.  Three implementations are given in
this file:

+ HyperBand
+ Grid Search
+ Random Search
"""

import random
from itertools import product
import sqlite3
import json
import time
import shutil

import os
from argparse import Namespace
import torch

import torch
import os
import pickle
import time
import random

def sync_parameters(args, rank, search_space, agent_class):
    """
    Synchronizes the parameters between the ranks of a process that is performing
    a single model training run.

    Arguments

    args : argparse.Namespace
        Arguments parsed by an argparse.ArgumentParser.  The arguments parsed which are to
        be used in the search_space should be lists of values which the arguments can take.
        The other arguments should be singletons which will be used by the script.
    rank : int
        Rank of this process in the process world
    search_space : list[str, ...]
        List of keys from the args which the hyperparameter search agent searches over.
        These are the parameters which need to be synchronized between process ranks.
    agent_class : object
        One of the classes in this file (or a custom class that you define) whose parameters
        will be synchronized across the ranks.

    Returns

    hyperparameter search agent class instance
    """
    # the path to the hyperparameter search directory
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

    # load the mutual file which holds the hyperparameters
    with open(os.path.join(path, f'{agree}/hparams.pkl'), 'rb') as fp:
        agent = pickle.load(fp)
    # return the synchronized agent object
    return agent

class DistributedAsynchronousHyperBand:
    """
    Distributed Asynchronous Hyperparameter Search object that implements the HyperBand
    algorithm https://arxiv.org/abs/1603.06560.  

    Attributes

    level : int
        The level of the hyperband search this instance is working on.  Models which are more successful
        at optimizing the metric than another in a pair of models compared move to the next (higher)
        level.  All models start at level 0.  The models which lose the comparison cease their training.
    schedule : list[int, ...]
        The schedule is a list that defines the number of epochs that take place at each level
    root : str
        The path to the directory which holds the hyperparameter search data
    search_space : list[str, ...]
        The names of the hyperparameters which are being searched over
    args : argparse.Namespace   
        The arguments parsed by the calling process that generated this hyperparameter search.  After
        a combination is chosen these will be replaced by the arguments corresponding to this combination.
        If the run has not been initialized, these arguments will be used to initialize the run.  If
        the run has already been initialized the passed arguments which are part of the search space
        will be ignored and a combination from the search space will be assigned.
    index : int
        The unique integer index for this combination of hyperparameters
    combination : tuple[any, ..]
        The combination of hyperparameters from the search space which correspond to self.index
    path : str
        The path of the checkpoint for this run


    Methods

    to_namespace
        updates the args argparse.Namespace by replacing the search space argument values with the
        values corresponding to the hyperparameters for this combination
    save_checkpoint
        Save a checkpoint of a model to the checkpoint path associated with the current combination
        of hyperparameters
    finish_combination
        Mark this agent's current run as finished - if this is not a terminal run in the hyperband
        search then the run is promoted to the next tier
    get_new_combination
        If a current combination is assigned, then that combination is finished and a new combination
        is retrieved, otherwise a new combination is assigned from the lowest available tier.
    get_path
        Returns the checkpoint path for the current hyperparameter combination.

    """


    def __init__(self, root, search_space, args, k=None, epoch_schedule=None):
        """
        TODO: support both minimizing and maximizing metric
        TODO: pass stragglers to the next level if they cannot otherwise be promoted
        """

        self.level = None
        self.schedule = epoch_schedule

        self.root = root
        self.search_space = search_space
        self.args = args


        # this is always just going to store the output from the dataset
        self.index = None
        self.combination = None
        self.path = None

        if not os.path.isdir(root):
            os.mkdir(root)

        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()

        try:
            cur.execute("CREATE TABLE todo0(num, hparams)")
            cur.execute("CREATE TABLE running0(num, hparams, chkpt)")
            cur.execute("CREATE TABLE finished0(num, hparams, chkpt, metric)")
            cur.execute("CREATE TABLE terminated0(num, hparams, chkpt, metric)")

            arg_list = tuple([vars(args)[key] for key in search_space])
            arg_product = random.choices(list(product(*arg_list)), k=k)

            for i, args in enumerate(arg_product):
                s = json.dumps({key: value for key, value in zip(search_space, args)})
                cur.execute(f"INSERT INTO todo0 VALUES ({i}, '{s}')")

        except Exception as e:
            pass

        con.commit()
        cur.close()

        self.get_new_combination()

    def to_namespace(self, combination):
        try:
            combination = json.loads(combination)
        except TypeError as e:
            print('no combination left to load...')
            return None
        args = vars(self.args)
        args.update(combination)
        return Namespace(**args)
    

    def save_checkpoint(self, states):
        with open(self.path, 'wb') as fp:
            torch.save(states, fp)


    def finish_combination(self, metric_value):
        combination = self.combination
        combination = json.dumps(self.combination)
        # access the shared file registry
        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()
        # remove the key from running
        cur.execute(f"DELETE FROM running{self.level} WHERE num = {self.index}")

        # add the key to the finished table
        cur.execute(f"INSERT INTO finished{self.level} VALUES"
            f"({self.index}, '{combination}', '{self.path}', {metric_value})")
        
        # compute winner if possible at current level
        res = cur.execute(f"SELECT * FROM todo{self.level}")
        res = res.fetchall()
            
        if len(res) > 1:
            a, b = res.pop(0), res.pop(0)

            win_index, win_comb, wp, wm = max([a, b], key=lambda k: k[-1])
            lose_index, lose_comb, lp, lm = min([a, b], key=lambda k: k[-1])
            
            cur.execute(f"INSERT INTO terminated{self.level} VALUES"
                        f"({lose_index}, '{lose_comb}', '{lp}', {lm})")
            cur.execute(f"INSERT INTO finished{self.level} VALUES"
                        f"({win_index}, '{win_comb}', '{wp}', {wm})")

            self.level = self.level + 1

            try:
                res = cur.execute(f"SELECT * FROM todo{self.level}")
                res = res.fetchall()
                # copy the checkpoint so we can load from it in the next iteration and continue training
                win_path = self.get_path(win_comb)
                shutil.copyfile(wp, win_path)
                cur.execute(f"INSERT INTO todo{self.level} VALUES"
                        f"({win_index}, '{win_comb}', {win_path}, {wm})")
            except Exception as e:
                win_path = self.get_path(win_comb)
                shutil.copyfile(wp, win_path)
                cur.execute(f"CREATE TABLE todo{self.level}(num, hparams, chkpt, metric)")
                cur.execute(f"INSERT INTO todo{self.level} VALUES"
                            f"({win_index}, '{win_comb}', {win_path}, {wm})")
        self.combination = None
        con.commit()
        cur.close()


    def get_new_combination(self, metric_value=None):
        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()
        # if has a combination:
        if self.combination is not None:
            self.finish_combination(metric_value)
            
        # select the lowest key with todos
        i = 0
        while i < 999:
            try:
                res = cur.execute(f"SELECT * FROM todo{i}")
                res = res.fetchall()
            except Exception as e:
                break
            i += 1

        i -= 1
        res = cur.execute(f"SELECT * FROM todo{i}")
        res = res.fetchall()

        if res:
            # assign variables for new combination
            if i > 0:
                self.index, self.combination, self.path, self.metric = res.pop(0)
                # mark the new combination as in progress
                cur.execute(f"DELETE FROM todo{i} WHERE num = {self.index}")
                cur.execute(f"INSERT INTO running{i} VALUES"
                            f"({self.index}, '{self.combination}', '{self.path}', {self.metric})")
            else:
                self.index, self.combination = res.pop(0)
                self.path = self.get_path(self.combination)
                # mark the new combination as in progress
                cur.execute(f"DELETE FROM todo{i} WHERE num = {self.index}")
                cur.execute(f"INSERT INTO running{i} VALUES"
                            f"({self.index}, '{self.combination}', '{self.path}')")
                
            con.commit()
            cur.close()
            return
        # if no todos remaining
        # select lowest key in progress
        i = 0
        while i < 999:
            try:
                res = cur.execute(f"SELECT * FROM running{i}")
                res = res.fetchall()
            except Exception as e:
                break
            i += 1

        i -= 1
        res = cur.execute(f"SELECT * FROM running{i}")
        res = res.fetchall()

        if res:
            if i > 0:
                self.index, self.combination, self.path, self.metric = res.pop(0)
                # mark the new combination as in progress
                cur.execute(f"INSERT INTO running{i} VALUES"
                            f"({self.index}, '{self.combination}', '{self.path}', {self.metric})")
            else:
                self.index, self.combination, self.path = res.pop(0)

            con.commit()
            cur.close()
            return
        
        self.combination = None
        con.commit()
        cur.close()


    def get_path(self, combination):
        if os.path.isdir(os.path.join(self.root, 'checkpoints')):
            if os.path.isdir(os.path.join(self.root, f'checkpoints/{self.level}')):
                i = 0
                while i < 999:
                    fp_path = os.path.join(self.root, f'checkpoints/{self.level}/{"_".join([str(json.loads(combination)[key]) for key in json.loads(combination)]) + f"_{i}.pt"}')
                    if os.path.isfile(fp_path):
                        i += 1
                    else:
                        with open(fp_path, 'wb') as fp:
                            pass
                        return fp_path
                raise ValueError("could not find file path suitable for this hyperparameter combination - ran out of possible paths")
            else:
                os.mkdir(os.path.join(self.root, f'checkpoints/{self.level}'))
                return self.get_path(combination)
        else:
            os.mkdir(os.path.join(self.root, 'checkpoints'))
            return self.get_path(combination)



class DistributedAsynchronousGridSearch:
    """
    Distributed Asynchronous Hyperparameter Search object that implements a Grid Search 

    Attributes

    root : str
        The path to the directory which holds the hyperparameter search data
    search_space : list[str, ...]
        The names of the hyperparameters which are being searched over
    args : argparse.Namespace   
        The arguments parsed by the calling process that generated this hyperparameter search.  After
        a combination is chosen these will be replaced by the arguments corresponding to this combination.
        If the run has not been initialized, these arguments will be used to initialize the run.  If
        the run has already been initialized the passed arguments which are part of the search space
        will be ignored and a combination from the search space will be assigned.
    index : int
        The unique integer index for this combination of hyperparameters
    combination : tuple[any, ..]
        The combination of hyperparameters from the search space which correspond to self.index
    path : str
        The path of the checkpoint for this run


    Methods

    to_namespace
        updates the args argparse.Namespace by replacing the search space argument values with the
        values corresponding to the hyperparameters for this combination
    save_checkpoint
        Save a checkpoint of a model to the checkpoint path associated with the current combination
        of hyperparameters
    finish_combination
        Mark this agent's current run as finished
    get_new_combination
        If a current combination is assigned, then that combination is finished and a new combination
        is retrieved, otherwise a new combination is assigned from the lowest available tier.
    get_path
        Returns the checkpoint path for the current hyperparameter combination.

    """


    def __init__(self, root, search_space, args, k=None, epoch_schedule=None):
        """

        """

        self.root = root
        self.search_space = search_space
        self.args = args

        # this is always just going to store the output from the dataset
        self.index = None
        self.combination = None
        self.path = None
        self.chkpt = None

        if not os.path.isdir(root):
            os.mkdir(root)

        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()

        try:
            cur.execute("CREATE TABLE todo(num, hparams)")
            cur.execute("CREATE TABLE running(num, hparams, chkpt)")
            cur.execute("CREATE TABLE finished(num, hparams, chkpt, metric)")

            arg_list = tuple([vars(args)[key] for key in search_space])
            arg_product = list(product(*arg_list))

            for i, args in enumerate(arg_product):
                s = json.dumps({key: value for key, value in zip(search_space, args)})
                cur.execute(f"INSERT INTO todo VALUES ({i}, '{s}')")

        except Exception as e:
            print(e)

        con.commit()
        cur.close()

        self.get_new_combination()


    def to_namespace(self, combination):
        try:
            combination = json.loads(combination)
        except TypeError as e:
            print('no combination left to load...')
            return None

        args = vars(self.args)
        args.update(combination)
        return Namespace(**args)
    

    def save_checkpoint(self, states):
        with open(self.path, 'wb') as fp:
            torch.save(states, fp)


    def finish_combination(self, metric_value):
        combination = self.combination
        combination = json.dumps(self.combination)
        # access the shared file registry
        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()
        # remove the key from 'in_progress'
        cur.execute(f"DELETE FROM running WHERE num = {self.index}")

        print(f"({self.index}, '{combination}', '{self.path}', {metric_value})")

        # add the key to the finished table
        cur.execute(f"INSERT INTO finished VALUES"
            f"({self.index}, '{combination}', '{self.path}', {metric_value})")
        
        con.commit()
        cur.close()


    def get_new_combination(self, metric_value=None):
        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()
        # if has a combination:
        if self.combination is not None:
            assert metric_value is not None, "metric value cannot be none when marking a run as finished"
            # mark current running combination as finished
            cur.execute(f"DELETE FROM running WHERE num = {self.index}")
            cur.execute(f"INSERT INTO completed VALUES"
                        f"({self.index}, '{self.combination}', {metric_value})")
            
        res = cur.execute("SELECT * FROM todo")
        res = res.fetchall()
            
        if res:
            # assign variables for new combination
            self.index, self.combination = res.pop(0)

            self.path = self.get_path(self.combination)
            # mark the new combination as in progress
            cur.execute(f"DELETE FROM todo WHERE num = {self.index}")
            cur.execute(f"INSERT INTO running VALUES"
                        f"({self.index}, '{self.combination}', '{self.path}')")
            con.commit()
            cur.close()
            return
        # if no todos remaining
        # select lowest key in progress
        res = cur.execute("SELECT * FROM running")
        res = res.fetchall()
        if res:
            # assign variables for new combination
            self.index, self.combination, self.path = res.pop(0)

            con.commit()
            cur.close()
            return
        
        con.commit()
        cur.close()

        self.combination = None


    def get_path(self, combination):
        if os.path.isdir(os.path.join(self.root, 'checkpoints')):
            if os.path.isdir(os.path.join(self.root, f'checkpoints/')):
                i = 0
                while i < 999:
                    fp_path = os.path.join(self.root, f'checkpoints/{"_".join([str(json.loads(combination)[key]) for key in json.loads(combination)]) + f"_{i}.pt"}')
                    if os.path.isfile(fp_path):
                        i += 1
                    else:
                        with open(fp_path, 'wb') as fp:
                            pass
                        return fp_path
                raise ValueError("could not find file path suitable for this hyperparameter combination - ran out of possible paths")
            else:
                os.mkdir(os.path.join(self.root, f'checkpoints/'))
                return self.get_path(combination)
        else:
            os.mkdir(os.path.join(self.root, 'checkpoints'))
            return self.get_path(combination)
        

class DistributedAsynchronousRandomSearch:
    """
    Distributed Asynchronous Hyperparameter Search object that implements a Grid Search 

    Attributes

    root : str
        The path to the directory which holds the hyperparameter search data
    search_space : list[str, ...]
        The names of the hyperparameters which are being searched over
    args : argparse.Namespace   
        The arguments parsed by the calling process that generated this hyperparameter search.  After
        a combination is chosen these will be replaced by the arguments corresponding to this combination.
        If the run has not been initialized, these arguments will be used to initialize the run.  If
        the run has already been initialized the passed arguments which are part of the search space
        will be ignored and a combination from the search space will be assigned.
    index : int
        The unique integer index for this combination of hyperparameters
    combination : tuple[any, ..]
        The combination of hyperparameters from the search space which correspond to self.index
    path : str
        The path of the checkpoint for this run


    Methods

    to_namespace
        updates the args argparse.Namespace by replacing the search space argument values with the
        values corresponding to the hyperparameters for this combination
    save_checkpoint
        Save a checkpoint of a model to the checkpoint path associated with the current combination
        of hyperparameters
    finish_combination
        Mark this agent's current run as finished
    get_new_combination
        If a current combination is assigned, then that combination is finished and a new combination
        is retrieved, otherwise a new combination is assigned from the lowest available tier.
    get_path
        Returns the checkpoint path for the current hyperparameter combination.

    """


    def __init__(self, root, search_space, args):
        """

        """

        self.root = root
        self.search_space = search_space
        self.args = args


        # this is always just going to store the output from the dataset
        self.index = None
        self.combination = None
        self.path = None
        self.chkpt = None

        if not os.path.isdir(root):
            os.mkdir(root)

        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()

        try:
            cur.execute("CREATE TABLE todo(num, hparams)")
            cur.execute("CREATE TABLE running(num, hparams, chkpt)")
            cur.execute("CREATE TABLE finished(num, hparams, chkpt, metric)")

            arg_list = tuple([vars(args)[key] for key in search_space])
            arg_product = list(product(*arg_list))

            for i, args in enumerate(arg_product):
                s = json.dumps({key: value for key, value in zip(search_space, args)})
                cur.execute(f"INSERT INTO todo VALUES ({i}, '{s}')")

        except Exception as e:
            pass

        con.commit()
        cur.close()

        self.get_new_combination()


    def to_namespace(self, combination):
        try:
            combination = json.loads(combination)
        except TypeError as e:
            print('no combination left to load...')
            return None
        args = vars(self.args)
        args.update(combination)
        return Namespace(**args)
    

    def save_checkpoint(self, states):
        with open(self.path, 'wb') as fp:
            torch.save(states, fp)


    def finish_combination(self, metric_value):
        combination = self.combination
        combination = json.dumps(self.combination)
        # access the shared file registry
        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()
        # remove the key from 'in_progress'
        cur.execute(f"DELETE FROM running WHERE num = {self.index}")

        # add the key to the finished table
        cur.execute(f"INSERT INTO finished VALUES"
            f"({self.index}, '{combination}', '{self.path}', {metric_value})")
        
        con.commit()
        cur.close()


    def get_new_combination(self, metric_value=None):
        con = sqlite3.connect(os.path.join(self.root, 'registry.db'))
        cur = con.cursor()
        # if has a combination:
        if self.combination is not None:
            assert metric_value is not None, "metric value cannot be none when marking a run as finished"
            # mark current running combination as finished
            cur.execute(f"DELETE FROM running WHERE num = {self.index}")
            cur.execute(f"INSERT INTO completed VALUES"
                        f"({self.index}, '{self.combination}', {metric_value})")
            
        res = cur.execute("SELECT * FROM todo")
        res = res.fetchall()
            
        if res:
            # assign variables for new combination
            self.index, self.combination = res.pop(random.choice(list(range(len(res)))))

            self.path = self.get_path(self.combination)
            # mark the new combination as in progress
            cur.execute(f"DELETE FROM todo WHERE num = {self.index}")
            cur.execute(f"INSERT INTO running VALUES"
                        f"({self.index}, '{self.combination}', '{self.path}')")
            con.commit()
            cur.close()
            return
        # if no todos remaining
        # select lowest key in progress
        res = cur.execute("SELECT * FROM running")
        res = res.fetchall()
        if res:
            # assign variables for new combination
            self.index, self.combination, self.path = res.pop(0)

            con.commit()
            cur.close()
            return
        
        con.commit()
        cur.close()

        self.combination = None


    def get_path(self, combination):
        if os.path.isdir(os.path.join(self.root, 'checkpoints')):
            if os.path.isdir(os.path.join(self.root, f'checkpoints/')):
                i = 0
                while i < 999:
                    fp_path = os.path.join(self.root, f'checkpoints/{"_".join([str(json.loads(combination)[key]) for key in json.loads(combination)]) + f"_{i}.pt"}')
                    if os.path.isfile(fp_path):
                        i += 1
                    else:
                        with open(fp_path, 'wb') as fp:
                            pass
                        return fp_path
                raise ValueError("could not find file path suitable for this hyperparameter combination - ran out of possible paths")
            else:
                os.mkdir(os.path.join(self.root, f'checkpoints/'))
                return self.get_path(combination)
        else:
            os.mkdir(os.path.join(self.root, 'checkpoints'))
            return self.get_path(combination)