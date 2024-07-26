import numpy as np
from mct.utils import *


def main():
    # 0. for each kind of embedding:
    # it is not a good idea to run this over the same embedding multiple times
    for embed in ['DINO', 'CLIP', 'EsViT', 'SwAV', 'MAE']:
        try:
            print(embed)
            np.random.seed(13)
            # 1. filter all but 1010 most common classes from the inat2017
            with open(f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/statistical distances/{embed}_iNat2017_train.ds', 'rb') as fp:
                x_train_embeds, x_train_labels = pickle.load(fp)
            # 2. select a stratified subset of the dataset to be the validation set
            unique, counts = np.unique(x_train_labels, return_counts=True)
            count_per_class = .1657262632 * counts # the validation fraction used in the original challenge
            # it's not filtered or anything so as long as the size is the same it shouldn't matter.  We're not going to report this number
            # ok, but this is not exactly 1% we will have some rounding to do here
            count_per_class = cascade_round(count_per_class)

            mask = np.hstack([np.random.choice(np.where(x_train_labels == l)[0], count_per_class[l], replace=False)
                                for l in unique])

            idx = mask
            
            unlb_idx = list(set(list(range(x_train_labels.shape[0]))) - set(idx))
            val_embeds = x_train_embeds[idx]
            val_labels = x_train_labels[idx]

            train_embeds = x_train_embeds[unlb_idx]
            train_labels = x_train_labels[unlb_idx]
            # 3. save the validation set and the original filtered set minus the validation set as val and train
            with open(f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/statistical distances/{embed}_iNat2017_train.ds', 'wb') as fp:
                pickle.dump((train_embeds, train_labels), fp)
            
            with open(f'/ourdisk/hpc/ai2es/jroth/AI2ES_DL_Torch/statistical distances/{embed}_iNat2017_val.ds', 'wb') as fp:
                pickle.dump((val_embeds, val_labels), fp)
        except:
            pass
        

if __name__ == "__main__":
    main()