import random
from itertools import product
import shutil
import os
import pickle
import torch
import time
import fcntl
from argparse import Namespace
import math

class DistributedAsynchronousHyperBand():


    def __init__(self, root, search_space, args, k=None, epoch_schedule=None):
        """
        TODO: support both minimizing and maximizing metric
        TODO: pass stragglers to the next level if they cannot otherwise be promoted
        """

        self.root = root
        self.combination = None
        self.search_space = search_space
        self.args = args
        self.level = None
        self.lockfile = None
        self.path = None
        self.schedule = epoch_schedule

        if os.path.isdir(root):
            time.sleep(random.randrange(0, 60)) # just wait a minute, because if this is the first group of processes submitted then 
            start = time.time()
            fpr = self.access_shared_file('rb') # access the shared file registry
            registry = pickle.load(fpr)
            self.close_shared_file(fpr)
            # fetch a hyperparameter combination at the lowest tier to work on
            # mark the parameter combination as in progress
            registry = self.get_new_combination(registry)
            # if registry was written to after reading raise an error - acceptable at this point to just restart the object
            if os.path.getctime(os.path.join(self.root, 'registry.pkl')) > start:
                raise SystemError('Write after Read')
            else:
                fp = self.access_shared_file()
                pickle.dump(registry, fp)
                self.close_shared_file(fp) # fine to release the lock after writing, but not after reading
        else:
            assert k is not None, "k cannot be none when initializing the search parameters"
            assert epoch_schedule is not None, "please define an epoch schedule"
            assert len(epoch_schedule) >= (math.log2(k) + 1), "please define an epoch schedule"

            arg_list = tuple([vars(args)[key] for key in search_space])
            arg_product = list(product(*arg_list))
            combination_sample = random.choices(arg_product, k=k)
            # write the combinations to a file
            self.combination = combination_sample.pop(-1)
            self.level = 0
            os.mkdir(root)
            fp = self.access_shared_file()
            self.path = self.get_path(self.combination)
            registry = {
                        0 : {
                            'todo': [(p, None, None) for p in combination_sample],
                            'in_progress': [(self.combination, self.path, None)],
                            'finished': [],
                            'terminated': [],
                            },
                        'completed_steps': []
                        }
            
            pickle.dump(registry, fp)
            self.close_shared_file(fp)

    def to_namespace(self, combination):
        args = vars(self.args)
        args.update({s: c for (s, c) in zip(self.search_space, combination)})
        args.update({'iters': self.schedule[self.level]})
        return Namespace(**args)

    def save_checkpoint(self, states):
        torch.save(states, self.path)


    def access_shared_file(self, mode='wb'):
        # access the shared registry that records the search progress
        """
        registry format

        tier : {
            todo: [combinations ...]
            in_progress: [combinations ...]
            finished: [(combination, checkpoint_path, metric_value)]
        }
        """
        fp = open(os.path.join(self.root, 'registry.pkl'), mode)

        if mode == 'wb':
            fcntl.flock(fp, fcntl.LOCK_EX)

        # acquire the lock
        # return the fp
        return fp

    def close_shared_file(self, fp):
        # ends this process' access to the shared file
        # release the lock
        fcntl.flock(fp, fcntl.LOCK_UN)
        # close the file
        fp.close()


    def finish_combination(self, metric_value):
        print('file')
        start = time.time()
        fpr = self.access_shared_file('rb') # access the shared file registry

        registry = pickle.load(fpr)
        self.close_shared_file(fpr)
        
        print('modifying registry')
        registry[self.level]['finished'] += [(self.combination, self.path, metric_value)]
        # remove the key from 'in_progress'
        try:
            idx = registry[self.level]['in_progress'].index((self.combination, self.path, None))
            registry[self.level]['in_progress'].pop(idx)
        except:
            print(registry[self.level]['in_progress'])
        # compute winner if possible at current level
        if len(registry[self.level]['finished']) > 1:                
            a, b = registry[self.level]['finished'].pop(0), registry[self.level]['finished'].pop(0)
            win_comb, wp, wm = max([a, b], key=lambda k: k[-1])
            lose_comb, lp, lm = min([a, b], key=lambda k: k[-1])
            registry[self.level]['terminated'].append((lose_comb, lp, lm))
            registry['completed_steps'].append((self.level, win_comb, wp, wm))
            self.level = self.level + 1
            if registry.get(self.level):
                # copy the checkpoint so we can load from it in the next iteration and continue training
                win_path = self.get_path(win_comb)
                shutil.copyfile(self.path, win_path)
                registry[self.level]['todo'].append((win_comb, win_path, None))
            else:
                win_path = self.get_path(win_comb)
                shutil.copyfile(self.path, win_path)
                registry[self.level] = {
                                        'todo': [(win_comb, win_path, None)],
                                        'in_progress': [],
                                        'finished': [],
                                        'terminated': [],
                                        }

        # if registry was written to after reading raise an error - not really acceptable at this point to just restart the object
        if os.path.getctime(os.path.join(self.root, 'registry.pkl')) > start:
            # so let's just wait up to a minute and try to save again
            time.sleep(random.randrange(0, 60))
            self.finish_combination(metric_value)
        else:
            fp = self.access_shared_file()
            print('write')
            pickle.dump(registry, fp)
            print('close write')
            self.close_shared_file(fp)


    def get_new_combination(self, registry, metric_value=None):
        # if has a combination:
        if self.combination is not None:
            assert metric_value is not None, "metric value cannot be none when marking a run as finished"
            # mark current 'in_progress' combination as finished
            registry[self.level]['finished'] += [(self.combination, self.path, metric_value)]
            # remove the key from 'in_progress'
            try:
                idx = registry[self.level]['in_progress'].index((self.combination, self.path, None))
                registry[self.level]['in_progress'].pop(idx)
            except:
                print(registry[self.level]['in_progress'])
            # compute winner if possible at current level
            if len(registry[self.level]['finished']) > 1:                
                a, b = registry[self.level]['finished'].pop(0), registry[self.level]['finished'].pop(0)
                win_comb, wp, wm = max([a, b], key=lambda k: k[-1])
                lose_comb, lp, lm = min([a, b], key=lambda k: k[-1])
                registry[self.level]['terminated'].append((lose_comb, lp, lm))
                registry['completed_steps'].append((self.level, win_comb, wp, wm))
                self.level = self.level + 1
                if registry.get(self.level):
                    win_path = self.get_path(win_comb)
                    shutil.copyfile(self.path, win_path)
                    registry[self.level]['todo'].append((win_comb, win_path, None))
                else:
                    win_path = self.get_path(win_comb)
                    shutil.copyfile(self.path, win_path)
                    registry[self.level] = {
                                            'todo': [(win_comb, win_path, None)],
                                            'in_progress': [],
                                            'finished': [],
                                            'terminated': [],
                                            }
            
        # select the lowest key with todos
        for key in range(len(registry)):
            if registry[key]['todo']:
                self.level = key
                # assign variables for new combination
                self.combination, p, _ = registry[key]['todo'].pop(0)

                self.path = p if p is not None else self.get_path(self.combination)
                # mark the new combination as in progress
                registry[key]['in_progress'] += [(self.combination, self.path, None)]
                return registry
        # if no todos remaining
        # select lowest key in progress
        for key in range(len(registry)):
            if registry[key]['in_progress']:
                self.level = key
                # assign variables for new combination
                self.combination, p, _ = registry[key]['in_progress'].pop(0)
                self.path = p if p is not None else self.get_path(self.combination)
                # mark the new combination as in progress
                registry[key]['in_progress'] += [(self.combination, self.path, None)]
                return registry
        self.combination = None
        return registry


    def get_path(self, combination):
        if os.path.isdir(os.path.join(self.root, 'checkpoints')):
            if os.path.isdir(os.path.join(self.root, f'checkpoints/{self.level}')):
                i = 0
                while i < 999:
                    fp_path = os.path.join(self.root, f'checkpoints/{self.level}/{"_".join([str(c) for c in combination]) + f"_{i}.pt"}')
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


class DistributedAsynchronousGridSearch():


    def __init__(self, root, search_space, args, k=None, epoch_schedule=None):
        """
        TODO: support both minimizing and maximizing metric
        TODO: pass stragglers to the next level if they cannot otherwise be promoted
        TODO: remove rank promotion
        TODO: remove random selection
        """

        self.root = root
        self.combination = None
        self.search_space = search_space
        self.args = args
        self.path = None
        self.schedule = epoch_schedule

        if os.path.isdir(root):
            time.sleep(random.randrange(0, 60)) # just wait a minute, because if this is the first group of processes submitted then 
            start = time.time()
            fpr = self.access_shared_file('rb') # access the shared file registry
            registry = pickle.load(fpr)
            self.close_shared_file(fpr)
            # fetch a hyperparameter combination at the lowest tier to work on
            # mark the parameter combination as in progress
            registry = self.get_new_combination(registry)
            # if registry was written to after reading raise an error - acceptable at this point to just restart the object
            if os.path.getctime(os.path.join(self.root, 'registry.pkl')) > start:
                raise SystemError('Write after Read')
            else:
                fp = self.access_shared_file()
                pickle.dump(registry, fp)
                self.close_shared_file(fp) # fine to release the lock after writing, but not after reading
        else:
            arg_list = tuple([vars(args)[key] for key in search_space])
            combination_sample = list(product(*arg_list))
            # write the combinations to a file
            self.combination = combination_sample.pop(-1)
            os.mkdir(root)
            fp = self.access_shared_file()
            self.path = self.get_path(self.combination)
            registry = {
                        'todo': [(p, None, None) for p in combination_sample],
                        'in_progress': [(self.combination, self.path, None)],
                        'completed_steps': []
                        }
            
            pickle.dump(registry, fp)
            self.close_shared_file(fp)

    def to_namespace(self, combination):
        args = vars(self.args)
        args.update({s: c for (s, c) in zip(self.search_space, combination)})
        return Namespace(**args)

    def save_checkpoint(self, states):
        with open(self.path, 'wb') as fp:
            pickle.dump(states, fp)


    def access_shared_file(self, mode='wb'):
        # access the shared registry that records the search progress
        """
        registry format

        tier : {
            todo: [combinations ...]
            in_progress: [combinations ...]
            finished: [(combination, checkpoint_path, metric_value)]
        }
        """
        fp = open(os.path.join(self.root, 'registry.pkl'), mode)

        if mode == 'wb':
            fcntl.flock(fp, fcntl.LOCK_EX)

        # acquire the lock
        # return the fp
        return fp

    def close_shared_file(self, fp):
        # ends this process' access to the shared file
        # release the lock
        fcntl.flock(fp, fcntl.LOCK_UN)
        # close the file
        fp.close()


    def finish_combination(self, metric_value):
        print('file')
        start = time.time()
        fpr = self.access_shared_file('rb') # access the shared file registry

        registry = pickle.load(fpr)
        self.close_shared_file(fpr)
        
        print('modifying registry')
        registry['completed_steps'].append((self.combination, self.path, metric_value))
        # remove the key from 'in_progress'
        try:
            idx = registry['in_progress'].index((self.combination, self.path, None))
            registry['in_progress'].pop(idx)
        except:
            print(registry['in_progress'])

        # if registry was written to after reading raise an error - not really acceptable at this point to just restart the object
        if os.path.getctime(os.path.join(self.root, 'registry.pkl')) > start:
            # so let's just wait up to a minute and try to save again
            time.sleep(random.randrange(0, 60))
            self.finish_combination(metric_value)
        else:
            fp = self.access_shared_file()
            print('write')
            pickle.dump(registry, fp)
            print('close write')
            self.close_shared_file(fp)


    def get_new_combination(self, registry, metric_value=None):
        # if has a combination:
        if self.combination is not None:
            assert metric_value is not None, "metric value cannot be none when marking a run as finished"
            # mark current 'in_progress' combination as finished
            registry['completed_steps'].append((self.combination, self.path, metric_value))
            # remove the key from 'in_progress'
            try:
                idx = registry['in_progress'].index((self.combination, self.path, None))
                registry['in_progress'].pop(idx)
            except:
                print(registry['in_progress'])
            
        if registry['todo']:
            # assign variables for new combination
            self.combination, p, _ = registry['todo'].pop(0)

            self.path = p if p is not None else self.get_path(self.combination)
            # mark the new combination as in progress
            registry['in_progress'] += [(self.combination, self.path, None)]
            return registry
        # if no todos remaining
        # select lowest key in progress
        if registry['in_progress']:
            # assign variables for new combination
            self.combination, p, _ = registry['in_progress'].pop(0)
            self.path = p if p is not None else self.get_path(self.combination)
            # mark the new combination as in progress
            registry['in_progress'] += [(self.combination, self.path, None)]
            return registry
        self.combination = None
        return registry


    def get_path(self, combination):
        if os.path.isdir(os.path.join(self.root, 'checkpoints')):
            if os.path.isdir(os.path.join(self.root, f'checkpoints/')):
                i = 0
                while i < 999:
                    fp_path = os.path.join(self.root, f'checkpoints/{"_".join([str(c) for c in combination]) + f"_{i}.pt"}')
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
