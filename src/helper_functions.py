"""
Helper functions used at various points. 
"""


import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


#root = data_dir = '/scratch/nsk367/dm2gal/dat/' #overlying directory to data
data_dir = '/scratch/nsk367/pytorch-use/research/dm2gal/dat/sampling/' #overlying directory to data
processed_path = root = '/scratch/nsk367/pytorch-use/research/dm2gal/dat/processed/'
dark_matter_path = data_dir + 'darkmatter/' # always the input, nbody simulation with all particles tracked
galaxy_path = data_dir + 'hydro/' # htotal - h subhalo (basically )
subhalo_path = data_dir + 'subhalos/'



def get_bins(params, num_bins):
    sample_counts, bin_edges, _ = plt.hist(params,num_bins)
    return sample_counts, bin_edges

def get_bin_num(bin_edges,value):
    # Given a data point, return the number bin it is in!
    bin_edges = np.asarray(bin_edges)
    idx = (np.abs(bin_edges - value.item())).argmin()
    if value > bin_edges[idx]:
        return idx 
    else:
        if idx > 0:
            return idx - 1
        else: 
            return idx

# define the function that randomly rotate the data
def random_rotation(dataset,rotation):

    if   rotation==0:   return np.rot90(dataset, 1, (0,1))
    elif rotation==1:   return np.rot90(dataset, 1, (0,2))
    elif rotation==2:   return np.rot90(dataset, 1, (1,2))
    elif rotation==3:   return np.rot90(dataset, 2, (0,1))
    elif rotation==4:   return np.rot90(dataset, 3, (0,1))
    elif rotation==5:   return np.rot90(np.rot90(dataset, 2, (0,1)), 1, (0,2))
    elif rotation==6:   return np.rot90(np.rot90(dataset, 3, (0,1)), 1, (0,2))
    elif rotation==7:   return np.rot90(np.rot90(dataset, 2, (0,1)), 1, (1,2))
    elif rotation==8:   return np.rot90(np.rot90(dataset, 3, (0,1)), 1, (1,2))
    elif rotation==9:   return np.rot90(dataset, 2, (0,2))
    elif rotation==10:  return np.rot90(dataset, 3, (0,2))
    elif rotation==11:  return np.rot90(np.rot90(dataset, 2, (0,2)), 1, (0,1))
    elif rotation==12:  return np.rot90(np.rot90(dataset, 3, (0,2)), 1, (0,1))
    elif rotation==13:  return np.rot90(np.rot90(dataset, 2, (0,2)), 1, (1,2))
    elif rotation==14:  return np.rot90(np.rot90(dataset, 3, (0,2)), 1, (1,2))
    elif rotation==15:  return np.rot90(dataset, 2, (1,2))
    elif rotation==16:  return np.rot90(dataset, 3, (1,2))
    elif rotation==17:  return np.rot90(np.rot90(dataset, 2, (1,2)), 1, (0,1))
    elif rotation==18:  return np.rot90(np.rot90(dataset, 3, (1,2)), 1, (0,1))
    elif rotation==19:  return np.rot90(np.rot90(dataset, 2, (1,2)), 1, (0,2))
    elif rotation==20:  return np.rot90(np.rot90(dataset, 3, (1,2)), 1, (0,2))
    elif rotation==21:  return np.rot90(np.rot90(dataset, 1, (0,1)), 1, (1,2))
    elif rotation==22:  return np.rot90(np.rot90(dataset, 3, (0,1)), 3, (1,2))
    elif rotation==23:  return dataset


def get_xyz(files):
    xyz = np.array([f.split('_') for f in files],dtype=int)
    x = xyz[:,0]
    y = xyz[:,1]
    z = xyz[:,2]
    return x,y,z

def get_mode(z):
    """
    If coordinates are in a particular subsection, flag what mode this is for future reference.
    
    "train"
    
    
    test
    """
    if 64 <= z['x'] <= 932 and  64 <= z['y'] <= 932 and  64 <= z['z'] <= 932: # To create a test space of approximately 32 Mpc/h
        tag = 'test'

    elif 1000 <= z['x'] <= 1800 and  1000 <= z['y'] <= 1800 and  1000 <= z['z'] <= 1800: # To create a valid space of approximately 30 Mpc/h
        tag = 'valid'
    
    else: #TODO
        tag = 'train'
    
    return tag


def read_all_maps(seed=42,mode='train',realizations=40000):
    """ Read as many realizations as you can to get the average value of maps and shmaps, and get coordinate csvs,     """
    from pandas import read_csv
    coords = read_csv(data_dir + 'saved_coords.csv')
    
    coords['Coordinates'] = coords['x'].astype(int).astype(str) + '_' + coords['y'].astype(int).astype(str) + '_' + coords['z'].astype(int).astype(str) 
    coords['Mode'] = coords.apply(lambda z: get_mode(z),axis=1)
    train_coords = coords.loc[coords['Mode'] == 'train']
    train_coords = train_coords.sample(frac=1)
    train_coords.to_csv(root + 'train_coords.csv',index=False)
    valid_coords = coords.loc[coords['Mode'] == 'valid']
    valid_coords.to_csv(root + 'valid_coords.csv',index=False)
    test_coords = coords.loc[coords['Mode'] == 'test']
    test_coords.to_csv(root + 'test_coords.csv',index=False)

    size = 2*32 + 1

    files =  train_coords['Coordinates'].values[:realizations] #random list of files to use 
    maps = np.zeros(shape = (len(files),size,size,size))
    params = np.zeros(shape = (len(files),1))
    shmaps = np.zeros(shape = (len(files),size,size,size))

    # do a loop over all realizations
    count = 0
    for i in range(len(files)):
        dm_f = np.load(dark_matter_path + 'dm' + files[i] + '.npy')
        dmsh_f = np.load(subhalo_path + 'subhalo' + files[i] + '.npy')
        gal_f = np.load(galaxy_path + 'galaxies' + files[i] + '.npy')
        maps[i,:] = dm_f
        shmaps[i,:] = dmsh_f
        params[i,:] = gal_f

        count += 1

    
    print("nsamples obtained by ", mode, ' = ', count) #use this to confirm they are right!
    
    maps_mean, maps_std = np.mean(maps), np.std(maps)
    shmaps_mean, shmaps_std = np.mean(shmaps), np.std(shmaps)


    #save to a file 
    np.save(root +  'mean_maps.npy', maps_mean)
    np.save(root + 'std_maps.npy', maps_std)

    np.save(root +  'mean_shmaps.npy', shmaps_mean)
    np.save(root + 'std_shmaps.npy', shmaps_std)
    

    params_mean = 8.750728
    params_std = 0.60161114
    
    params = (params - params_mean) / params_std

    num_bins = 100
    bin_sample_counts, bin_edges = get_bins(params.flatten(), num_bins)
    bin_weights = 1./torch.Tensor(bin_sample_counts)
    train_targets = [get_bin_num(bin_edges,sample) for sample in params.flatten()]
    train_samples_weight = [bin_weights[bin_id] for bin_id in train_targets]

    train_samples_weight = np.array(train_samples_weight)
    
    np.save(root + "train_sample_weights.npy",train_samples_weight)



