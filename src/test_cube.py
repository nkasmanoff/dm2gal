"""

Load in model that performs best at reconstructiong the power spectrum in the validation space,
 and use it to reconstruct over the test cube space, which is larger and contains 10x more galaxies. 

After saved in processed dir, use a jupyter notebook to analyze.
"""
import Pk_library as PKL

from src.models.models import *
import dataloader as dataloader
#from src.utils.utils import *
from predict_stellar_mass import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from test_tube import HyperOptArgumentParser, SlurmCluster
from argparse import ArgumentParser, Namespace
import os
import random
import sys
import numpy as np
import pandas as pd


# load best

best_model_path = '../model/dm2gal_weights.ckpt' # '/projects/QUIJOTE/Noah/logs/final/lightning_logs/version_5167072/checkpoints/epoch=8.ckpt'
# in this repo version
#best_model_path = '../model/epoch=8.ckpt'


version = 'final'

#best_model_path.split('lightning_logs')[1][1:].split('/')[0]
#print(version)
#sys.exit()
model = Galaxy_Model.load_from_checkpoint(best_model_path)    #masking_model, using the latest version. 

model.eval()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("Device = ", device)

print("Loading in  best model...")
print("best model path = ", best_model_path)
root = '/scratch/nsk367/pytorch-use/research/dm2gal/dat/processed/'
data_dir = '/scratch/nsk367/pytorch-use/research/dm2gal/dat/sampling/' #overlying directory to data
dark_matter_path = data_dir + 'darkmatter/' # always the input, nbody simulation with all particles tracked
galaxy_path = data_dir + 'hydro/' # htotal - h subhalo (basically )
subhalo_path = data_dir + 'subhalos/'

test_coords = pd.read_csv(root + 'test_coords.csv')

print("loading in coords...")

#Load the actual files in here!

files = test_coords['Coordinates'].values #in test or validation mode, use everything to recreate our mini universe.
maps_mean = 5.20168457 #np.load(root + 'mean_maps.npy')
maps_std = 3.11864378 #np.load(root + 'std_maps.npy')
shmaps_mean = 0.41816423 #np.load(root + 'mean_shmaps.npy')
shmaps_std = 1.73515605 #np.load(root + 'std_shmaps.npy')


params_mean, params_std = 8.750728, 0.60161114 # only with galaxies > 1e8 solar masses 

coords_cube = test_coords[['x','y','z']]
max_coord = coords_cube.to_numpy().max()
min_coord = coords_cube.to_numpy().min()
cube_len = int(max_coord - min_coord + 1)
coords_cube = coords_cube - min_coord #whatever the offset is, which is 64 from central dataloader.  

print("Cube Length = ", cube_len)


cube_pred = np.zeros(shape = (cube_len,cube_len,cube_len)) 
cube_target = np.zeros(shape = (cube_len,cube_len,cube_len)) 

#benchmark done separately in hod.py

print("Beginning reconstruction")
for i in range(coords_cube.shape[0]):
    params = np.load(galaxy_path + 'galaxies' + files[i] + '.npy')  
    maps = np.load(dark_matter_path + 'dm' + files[i] + '.npy')
    shmaps  = np.load(subhalo_path + 'subhalo' + files[i] + '.npy')
    maps   = (maps   - maps_mean)/maps_std
    shmaps   = (shmaps   - shmaps_mean)/shmaps_std

    #params = (params - params_mean)/params_std  
    maps = torch.Tensor(np.stack((maps,shmaps)).reshape(1,2,65,65,65))
    predicted_mass = model.forward(maps.cuda()).cpu().item()
    predicted_mass = predicted_mass * params_std + params_mean
    predicted_mass = (10**predicted_mass) - 1

    x = int(coords_cube['x'].values[i])
    y = int(coords_cube['y'].values[i])
    z = int(coords_cube['z'].values[i])
    cube_pred[x,y,z] = predicted_mass

    target_mass = (10**params[0]) - 1
    cube_target[x,y,z] = target_mass


cube_pred = cube_pred.astype(np.float32)
cube_target = cube_target.astype(np.float32)

print("Target cube statistics:")
print("sum= ", cube_target.sum())
print("Std= ", cube_target.std())

print("Predicted cube statistics:")
print("sum= ", cube_pred.sum())
print("Std= ", cube_pred.std())


print("Saving Cubes.")
#save cube 
np.save(root + "test_cube_"+version+"_prediction.npy",cube_pred)
np.save(root + "test_cube_target.npy",cube_target)

print("Done! ")
