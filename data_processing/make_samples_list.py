import torch
import torchvision
import pickle

ds = torchvision.datasets.ImageFolder('/ourdisk/hpc/ai2es/datasets/LAION/LAION-400M/your_output_folder')

with open('/ourdisk/hpc/ai2es/jroth/400M_samples_list.pkl', 'wb') as fp:
    pickle.dump(ds.samples, fp)
