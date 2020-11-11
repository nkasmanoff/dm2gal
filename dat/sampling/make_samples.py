"""

Iterate over the density field files (darkmatter, subhalos, and hydro) and save in bite-sized samples to be fed into 
a machine learning model.


"""
#import dependencies

import numpy as np
import pandas as pd
import h5py
from itertools import product
import sys


# initialize

size = 32 #box size for input is 2 * size + 1. + 1 since the target is directly at the center. 

pos=list(np.arange(size,2000,1)) 
   


#all dark matter particles n body sim
dm_root = '../illustris_tng/darkmatter/' #subhalo root 
#dm_pfield = dm_root+'TNG100-1_z=0.0.hdf5'   #note z is in here. 
dm_pfield = dm_root + 'TNG100-1_CDM_z=0.0.hdf5'
delta_cdm = h5py.File(dm_pfield,'r')['delta_cdm'][:]
delta_cdm = np.log10(1 + delta_cdm) #normalize

print("DM done.")


#dark matter only subhalos
dmsh_root = '../illustris_tng/subhalos/' #subhalo root 
dmsh_pfield = dmsh_root+'subhalos_TNG100-1_z=0.0.hdf5'   #note z is in here. 
density_subhalo = h5py.File(dmsh_pfield,'r')['density'][:]
density_subhalo = np.log10(1 + density_subhalo) #normalize

print("DMSH done.")


# hydro galaxies 
galaxies_root = '../illustris_tng/hydro/' # galaxy root
gal_pfield = galaxies_root+'stellarmass_TNG100-1_z=0.0.hdf5'
density_gal = h5py.File(gal_pfield,'r')['density'][:]
density_gal = np.log10(1 + density_gal)

print("Galxies done.")


#iterate and save
ranges=list(product(pos,repeat=3))


data_dir = '../sampling/'
dark_matter_path = data_dir + 'darkmatter/'
galaxy_path = data_dir + 'hydro/'
subhalo_path = data_dir + 'subhalos/'

xs = np.zeros(len(ranges))
ys = np.zeros(len(ranges))
zs = np.zeros(len(ranges))


for i,ID in enumerate(ranges):
    box = str(ID[0])+'_'+str(ID[1])+'_'+str(ID[2])  # the actual grid space currently being observed. #001 to 2000,2000,2000?  
    galaxy_voxel  = density_gal[ID[0],ID[1],ID[2]]
    if galaxy_voxel > 8:  # Only saves samples if corresponding stellar mass > 1e8. 
        xs[i] = ID[0]
        ys[i] = ID[1]
        zs[i] = ID[2]
        dm_voxel = delta_cdm[ID[0]-size:ID[0]+size + 1,ID[1]-size:ID[1]+size+1,ID[2]-size:ID[2]+size+1] #covers a larger area than the others.
        subhalo_voxel =  density_subhalo[ID[0]-size:ID[0]+size + 1,ID[1]-size:ID[1]+size+1,ID[2]-size:ID[2]+size+1] 
        np.save(dark_matter_path + 'dm'+box +'.npy',dm_voxel) 
        np.save(galaxy_path + 'galaxies'+box+'.npy',np.array([galaxy_voxel])) 
        np.save(subhalo_path+ 'subhalo'+box+'.npy',subhalo_voxel) 

coords = pd.DataFrame() 
xs = xs[xs > 0] # In case I made this array too large, this removes all empty values.ß
ys = ys[ys > 0]
zs = zs[zs > 0]

coords['x'] = xs
del xs
coords['y'] = ys
del ys
coords['z'] = zs
del zs 
#save coordinates to read in later by dataloader 
coords.to_csv(data_dir + 'saved_coords.csv',index=False)
