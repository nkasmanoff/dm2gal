"""

"""

#import dependencies

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from src.helper_functions import *

#root = data_dir = '/scratch/nsk367/dm2gal/dat/' #overlying directory to data
data_dir = '../dat/sampling/' #overlying directory to data
processed_path = root = '../dat/processed/'
dark_matter_path = data_dir + 'darkmatter/' # always the input, nbody simulation with all particles tracked
galaxy_path = data_dir + 'hydro/' # htotal - h subhalo (basically )
subhalo_path = data_dir + 'subhalos/'



# Define Dataset 
class make_Dataset(Dataset):
    """
    How to load and read dataset. In this case, let's update where maps are read
    based on mode? 
    """
    def __init__(self, augment=True,mode='train', realizations=2048, seed = 42):
        #load in files from read maps

        file_csv = root + mode + "_coords.csv"
        if not(os.path.exists(file_csv)):  
            print("Reading all maps)")
            read_all_maps(seed=seed,mode=mode,realizations=realizations)
        self.files = pd.read_csv(file_csv) 
        self.augment = augment
        self.maps_mean = 5.20168457 #np.load(root + 'mean_maps.npy')
        self.maps_std = 3.11864378 #np.load(root + 'std_maps.npy')
        self.shmaps_mean = 0.41816423 #np.load(root + 'mean_shmaps.npy')
        self.shmaps_std = 1.73515605 #np.load(root + 'std_shmaps.npy')
        self.params_mean = 8.750728
        self.params_std = 0.60161114

        self.files = self.files['Coordinates'].values #random list of files to use 



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        maps = np.load(dark_matter_path + 'dm' + self.files[idx] + '.npy')
        shmaps = np.load(subhalo_path + 'subhalo' + self.files[idx] + '.npy')
        params = np.load(galaxy_path + 'galaxies' + self.files[idx] + '.npy')
        #now standardize, and save to root dir. 
        maps   = (maps   - self.maps_mean)/self.maps_std
        shmaps  = (shmaps - self.shmaps_mean) / self.shmaps_std
        params = (params - self.params_mean)/self.params_std

        if self.augment:
            rotation = np.random.choice(np.arange(24))
            maps = random_rotation(maps,rotation)
            shmaps = random_rotation(shmaps,rotation)

        maps = np.stack((maps,shmaps),axis=0) #0 for channel wise. 
        maps = torch.from_numpy(maps)
        params = torch.from_numpy(params)

        return maps, params   


    

# This function creates the different datasets
def create_datasets(realizations, batch_size, seed = 42, weighted_trainer = True):
    train_Dataset = make_Dataset(augment=True,mode='train',realizations=realizations,seed=seed)
    if weighted_trainer:
        train_samples_weight = np.load(root + "train_sample_weights.npy")
        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, train_Dataset.__len__())
        train_loader = DataLoader(train_Dataset, batch_size=batch_size,sampler=train_sampler)
    
    else: 
        train_loader = DataLoader(train_Dataset, batch_size=batch_size,shuffle=True)

    valid_Dataset = make_Dataset(augment=False,mode='valid',realizations=realizations,seed=seed)
    
    valid_loader  = DataLoader(dataset=valid_Dataset, batch_size=1, 
                               shuffle=False)


    return train_loader,valid_loader #test data treated entirely separately.  



if __name__ == '__main__':
    train_loader, valid_loader = create_datasets(realizations=40000,batch_size=16)
