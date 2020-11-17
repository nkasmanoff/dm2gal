"""

Halo Occupation Distribution baseline code. 


"""

#import dependencies
import numpy as np
import MAS_library as MASL
import sys,os,h5py,time
import units_library as UL
import matplotlib.pyplot as plt
import h5py

dims = 2048
MAS  = 'CIC'
#import illustris_python as il
delta = np.zeros((dims,dims,dims), dtype=np.float32)

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3

# read header
FILE_TYPE = '/hydro/' #or subhalos

#TODO: Before this, cp from fof_files into a new folder made @ ../dat/illustris_tng/hydro/groupcat/groups_099
# Once done, you can use TNG's illustris-python code to read in this data for hod computation


root = '../dat/illustris_tng/' + FILE_TYPE + 'groupcat/groups_099/'
basePath = '../dat/illustris_tng' + FILE_TYPE + 'groupcat'
# /scratch/gpfs/nk11/dm2gal/dat/illustris-data/hydrosim

# this part you'll need to download. 
#sys.path.append("/home/nk11")
sys.path.append('/scratch/nsk367/pytorch-use/research')

import illustris_python as il
import pandas as pd



f = h5py.File(root + os.listdir(root)[0],'r')

redshift = f['Header'].attrs[u'Redshift']
Omega_m  = f['Header'].attrs[u'Omega0']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
Nall     = f['Header'].attrs[u'Nsubgroups_Total']


r = il.groupcat.load(basePath,snapNum=99)

halo_mass = r['halos']['GroupMass'][r['subhalos']['SubhaloGrNr']]*1e10
n_subhalos = r['halos']['GroupNsubs'][r['subhalos']['SubhaloGrNr']]
galaxy_mass = r['subhalos']['SubhaloMassType'][:,4]*1e10


central_bool = np.zeros(shape = r['subhalos']['count'])
for i in range(r['halos']['count']):
    sh_central = r['halos']['GroupFirstSub'][i] 
    if sh_central == -1:
        pass
    central_bool[sh_central] = 1
    

def get_bins(params, num_bins):
    sample_counts, bin_edges, _ = plt.hist(params,num_bins)
    return sample_counts, bin_edges

sample_counts, bin_edges = get_bins(np.log10(halo_mass),num_bins=10)


def get_bin_num(bin_edges,value):
    # Given a data point, return the number bin it is in!
    bin_edges = np.asarray(bin_edges)
    idx = (np.abs(bin_edges - value.item())).argmin() #finding if it's between 
    if value > bin_edges[idx]:
        return idx 
    else:
        if idx > 0:
            return idx - 1
        else: 
            return idx
    
    
target_bins = np.array([get_bin_num(bin_edges,sample) for sample in np.log10(halo_mass)])

hod_df = pd.DataFrame()
hod_df['halo_mass'] = halo_mass
hod_df['target_bin'] = target_bins
hod_df['central_flag'] = central_bool
hod_df['galaxy_mass'] = galaxy_mass
hod_df['n_subhalos'] = n_subhalos
hod_df = hod_df.loc[hod_df['galaxy_mass'] > 0] #conditional 



# read header
FILE_TYPE = 'subhalos/' 

root = '../dat/illustris_tng/' + FILE_TYPE + 'groupcat/groups_099/'
basePath = '../dat/illustris_tng/' + FILE_TYPE + 'groupcat'

r = il.groupcat.load(basePath,snapNum=99)

sh_halomass = r['halos']['GroupMass'][r['subhalos']['SubhaloGrNr']]*1e10
n_nbody_subhalos = r['halos']['GroupNsubs'][r['subhalos']['SubhaloGrNr']]
pos = (r['subhalos']['SubhaloPos']/1e3).astype(np.float32)
sh_central_bool = np.zeros(shape = r['subhalos']['count'])
for i in range(r['halos']['count']):
    sh_central = r['halos']['GroupFirstSub'][i]
    if sh_central == -1:
        pass
    sh_central_bool[sh_central] = 1


target_bins = np.array([get_bin_num(bin_edges,sample) for sample in np.log10(sh_halomass)])


sh_df = pd.DataFrame()
sh_df['halo_mass'] = sh_halomass
sh_df['target_bin'] = target_bins
sh_df['target_bin'] = sh_df['target_bin'].apply(lambda z: z - 1 if z == 10 else z)
sh_df['central_flag'] = sh_central_bool
sh_df['n_subhalos'] = n_nbody_subhalos 
sh_df['parent_halo_idx'] = r['subhalos']['SubhaloGrNr']
sh_df['x'] = pos[:,0]
sh_df['y'] = pos[:,1]
sh_df['z'] = pos[:,2]


avg_hydro_satellites = hod_df.loc[hod_df['central_flag'] == 1].groupby('target_bin')['n_subhalos'].mean()

avg_nbody_subhalos = sh_df.loc[sh_df['central_flag'] == 1].groupby('target_bin')['n_subhalos'].mean()
avg_nbody_subhalos = avg_nbody_subhalos.reset_index()



nbody = []
hydro = []
for target_bin in range(50):
    nbody_count = sh_df.loc[sh_df['target_bin'] == target_bin].shape[0]
    hydro_count = hod_df.loc[hod_df['target_bin'] == target_bin].shape[0]
    
    #print(nbody_count, hydro_count)
    nbody.append(nbody_count)
    hydro.append(hydro_count)





sh_hod_df = pd.DataFrame()
for bin_central,g in sh_df.groupby(['target_bin','central_flag']):
    target_bin,central_bool = bin_central
    if central_bool == 0:
      #  print("satellite")
        average_hydro_bin_subhalos = avg_hydro_satellites.iloc[target_bin]
        average_nbody_bin_subhalos = avg_nbody_subhalos.loc[avg_nbody_subhalos['target_bin'] == target_bin]['n_subhalos'].values[0]
#    break
        gal_mass = hod_df.loc[(hod_df['central_flag'] == central_bool) & (hod_df['target_bin'] == target_bin)]['galaxy_mass'].sample(n=g.shape[0],replace=True).values * (average_hydro_bin_subhalos/average_nbody_bin_subhalos) * (hydro[target_bin] / nbody[target_bin])
    
    else: 
      #  print("central")
        gal_mass = hod_df.loc[(hod_df['central_flag'] == central_bool) & (hod_df['target_bin'] == target_bin)]['galaxy_mass'].sample(n=g.shape[0],replace=True).values
        
    g['galaxy_mass'] = gal_mass 
    sh_hod_df = pd.concat([sh_hod_df,g],axis=0)
    
print("Onto Pylians part...") 

mass = sh_hod_df['galaxy_mass'].values.astype(np.float32)
pos = sh_hod_df[['x','y','z']].values.astype(np.float32)

dims = 2048
MAS  = 'CIC'
#import illustris_python as il
delta = np.zeros((dims,dims,dims), dtype=np.float32)

U = UL.units();  rho_crit = U.rho_crit #h^2 Msun/Mpc^3

f = h5py.File(root + os.listdir(root)[0],'r')

redshift = f['Header'].attrs[u'Redshift']
Omega_m  = f['Header'].attrs[u'Omega0']
BoxSize  = f['Header'].attrs[u'BoxSize']/1e3  #Mpc/h
Omega_L  = f['Header'].attrs[u'OmegaLambda']
h        = f['Header'].attrs[u'HubbleParam']
Nall     = f['Header'].attrs[u'Nsubgroups_Total']

print("Creating delta field")
MASL.MA(pos, delta, BoxSize, MAS, mass)  #stars
M_total = np.sum(mass, dtype=np.float64)

# Now the region corresponding to where the test set is, all that's
# relevant from this method. 
benchmark_cube = delta[64:940,64:940,64:940]
np.save('../dat/processed/benchmark_cube.npy',benchmark_cube)
