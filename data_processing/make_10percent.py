import os
import shutil
import torch, torchvision


if __name__ == "__main__":
    with open('/ourdisk/hpc/ai2es/datasets/Imagenet/10percent.txt', 'r') as fp:
        for line in fp.readlines():
            c, i = line.strip().split('_') # class and identifier
            frompath = f'/ourdisk/hpc/ai2es/datasets/Imagenet/2012/train/{c}/{c}_{i}'
            todir = f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/10percent/{c}'
            topath = f'{todir}/{c}_{i}'
            if os.path.isdir(todir):
                shutil.copy(frompath, topath)
            else:
                os.mkdir(todir)
                shutil.copy(frompath, topath)

    with open('/ourdisk/hpc/ai2es/datasets/Imagenet/1percent.txt', 'r') as fp:
        for line in fp.readlines():
            c, i = line.strip().split('_') # class and identifier
            frompath = f'/ourdisk/hpc/ai2es/datasets/Imagenet/2012/train/{c}/{c}_{i}'
            todir = f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/1percent/{c}'
            topath = f'{todir}/{c}_{i}'
            if os.path.isdir(todir):
                shutil.copy(frompath, topath)
            else:
                os.mkdir(todir)
                shutil.copy(frompath, topath)
    
    dataset1 = torchvision.datasets.ImageNet('/ourdisk/hpc/ai2es/datasets/Imagenet/2012', split='train')
    dataset2 = torchvision.datasets.ImageFolder('/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/MCT/1percent/')

    for k1, k2 in zip(dataset1.class_to_idx.keys(), dataset2.class_to_idx.keys()):
        assert k1 == k2, f'{k1} and {k2} do not match for classes {dataset1.class_to_idx[k1]} and {dataset2.class_to_idx[k2]}'